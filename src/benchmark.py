#!/usr/bin/env python3
"""
Benchmark script for Qwen3-4B vLLM deployment on Modal.
Measures cold start, TPS, latency, and throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


# ============================================================
# Configuration
# ============================================================

DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
    "What are the key differences between AI and AGI?",
    "Describe the process of photosynthesis.",
    "Write a haiku about coding.",
    "Explain how vLLM achieves high throughput.",
    "What are the benefits of FP8 quantization?",
    "Write a short story about a robot learning to paint.",
    "Compare and contrast Modal vs AWS Lambda.",
    "Explain why transformer models need attention mechanisms.",
]

DEFAULT_CONCURRENCY = 10  # Reduced for budget safety
DEFAULT_MAX_TOKENS = 256  # Reduced for faster benchmarks
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_REQUESTS = 50  # Reduced for budget safety


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    endpoint: str
    num_requests: int = DEFAULT_NUM_REQUESTS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    concurrency: int = DEFAULT_CONCURRENCY
    output_file: str = "benchmark_results.json"
    measure_cold_start: bool = False
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BenchmarkConfig":
        """Create config from parsed command-line arguments."""
        prompts = cls._parse_prompts(args.prompts)
        return cls(
            endpoint=args.endpoint,
            num_requests=args.requests,
            max_tokens=args.max_tokens or DEFAULT_MAX_TOKENS,
            temperature=args.temperature or DEFAULT_TEMPERATURE,
            concurrency=args.concurrency,
            output_file=args.output,
            measure_cold_start=args.cold_start,
            prompts=prompts,
        )

    @staticmethod
    def _parse_prompts(prompts_arg: str | None) -> list[str]:
        """Parse comma-separated prompts from CLI argument."""
        if prompts_arg:
            return [p.strip() for p in prompts_arg.split(",")]
        return list(DEFAULT_PROMPTS)

    def validate(self) -> bool:
        """Validate configuration before running benchmark."""
        if not self.endpoint:
            rprint("[red]Error: No endpoint specified![/red]")
            rprint("Provide via --endpoint or MODAL_ENDPOINT in .env")
            return False

        # Ensure endpoint has correct path for OpenAI API
        if "/v1/chat/completions" not in self.endpoint:
            # Strip /health or /generate if present
            base = self.endpoint.rstrip("/")
            self.endpoint = f"{base}/v1/chat/completions"

        return True


# ============================================================
# Data Models
# ============================================================

@dataclass
class RequestResult:
    """Result from a single request."""
    success: bool
    latency_ms: float
    tokens: int


@dataclass
class BenchmarkResult:
    """Results from a complete benchmark run."""
    total_requests: int
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    cold_start_ms: float = 0.0

    # Computed properties
    @property
    def tps(self) -> float:
        """Tokens per second (throughput)."""
        return self.total_tokens / self.total_time_seconds if self.total_time_seconds > 0 else 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p50_latency_ms(self) -> float:
        """P50 latency (median)."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        mid = len(sorted_latencies) // 2
        if len(sorted_latencies) % 2 == 0:
            return (sorted_latencies[mid - 1] + sorted_latencies[mid]) / 2
        return sorted_latencies[mid]

    @property
    def p99_latency_ms(self) -> float:
        """P99 latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def requests_per_second(self) -> float:
        """Requests processed per second."""
        return self.successful_requests / self.total_time_seconds if self.total_time_seconds > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cold_start_ms": self.cold_start_ms,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": self.success_rate,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": self.total_tokens / self.successful_requests if self.successful_requests > 0 else 0,
            "total_time_seconds": self.total_time_seconds,
            "tps": self.tps,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "requests_per_second": self.requests_per_second,
        }


# ============================================================
# Cold Start Measurement
# ============================================================

async def measure_cold_start(endpoint: str) -> float:
    """
    Measure cold start time for the vLLM server.

    Cold start includes:
    - Container startup
    - Model loading into GPU memory
    - CUDA graph compilation
    - First completion (not just /v1/models)

    Returns:
        Cold start time in milliseconds, or -1 on failure
    """
    base_url = endpoint.replace("/v1/chat/completions", "")

    rprint("[bold cyan]Measuring cold start...[/bold cyan]")
    rprint("[dim]This measures container startup + model loading + first completion[/dim]\n")

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            # Wait for server to be ready using /v1/models
            response = await client.get(
                f"{base_url}/v1/models",
                timeout=120.0,
            )
            response.raise_for_status()

            # Now send a minimal completion request to measure true cold start
            # This ensures CUDA graphs, scheduler, and memory pools are warmed
            response = await client.post(
                endpoint,
                json={
                    "model": "Qwen/Qwen3-4B-Instruct-2507",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                },
                timeout=120.0,
            )
            response.raise_for_status()
    except httpx.TimeoutException:
        rprint("[red]Cold start timeout (server didn't respond within 120s)[/red]")
        return -1
    except Exception as e:
        rprint(f"[red]Cold start failed: {e}[/red]")
        return -1

    cold_start_ms = (time.perf_counter() - start) * 1000
    rprint(f"[green]Cold start: {cold_start_ms/1000:.1f}s ({cold_start_ms:.0f}ms)[/green]\n")

    return cold_start_ms


# ============================================================
# Request Handler
# ============================================================

class RequestHandler:
    """Handles HTTP requests for the benchmark."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "RequestHandler":
        self.client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.client:
            await self.client.aclose()

    async def make_request(self, prompt: str) -> RequestResult:
        """Make a single generation request to OpenAI-compatible endpoint."""
        start = time.perf_counter()
        try:
            response = await self.client.post(
                self.config.endpoint,
                json={
                    "model": "Qwen/Qwen3-4B-Instruct-2507",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
                timeout=120.0,  # Allow 2 minutes per request
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                # Count output tokens from the response
                usage = data.get("usage", {})
                tokens = usage.get("completion_tokens", 0)
                return RequestResult(success=True, latency_ms=latency_ms, tokens=tokens)
            else:
                return RequestResult(success=False, latency_ms=latency_ms, tokens=0)

        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000 if 'start' in locals() else 0
            return RequestResult(success=False, latency_ms=latency_ms, tokens=0)


# ============================================================
# Result Formatter
# ============================================================

class ResultFormatter:
    """Formats and displays benchmark results."""

    def __init__(self, console: Console):
        self.console = console

    def print_header(self, config: BenchmarkConfig) -> None:
        """Print benchmark header."""
        rprint("\n[bold cyan]Qwen3-4B vLLM Benchmark[/bold cyan]")
        rprint(f"[dim]Model: Qwen/Qwen3-4B-Instruct-2507[/dim]")
        rprint(f"[dim]Endpoint: {config.endpoint}[/dim]")
        rprint(f"[dim]Requests: {config.num_requests} | Concurrency: {config.concurrency}[/dim]\n")

    def print_cold_start(self, cold_start_ms: float) -> None:
        """Print cold start result."""
        if cold_start_ms > 0:
            rprint(f"[bold green]Cold Start: {cold_start_ms/1000:.1f}s ({cold_start_ms:.0f}ms)[/bold green]")
        elif cold_start_ms < 0:
            rprint("[bold red]Cold Start: Failed[/bold red]")

    def print_results(self, result: BenchmarkResult, num_requests: int) -> None:
        """Print beautiful benchmark results."""
        rprint("\n" + "=" * 60)
        rprint("[bold cyan]Benchmark Results[/bold cyan]")
        print("=" * 60)

        # Summary table
        table = Table(title="Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Requests", str(num_requests))
        table.add_row("Successful", f"[green]{result.successful_requests}[/green]")
        table.add_row("Failed", f"[red]{result.failed_requests}[/red]" if result.failed_requests > 0 else "[green]0[/green]")
        table.add_row("Success Rate", f"{result.success_rate:.1f}%")
        table.add_row("Total Tokens", f"[blue]{result.total_tokens:,}[/blue]")
        table.add_row("Total Time", f"[cyan]{result.total_time_seconds:.2f}s[/cyan]")

        self.console.print(table)

        # TPS metrics
        tps_table = Table(title="Throughput")
        tps_table.add_column("Metric", style="cyan")
        tps_table.add_column("Value", style="magenta")

        tps_color = "green" if result.tps >= 500 else "yellow" if result.tps >= 200 else "red"
        tps_emoji = "🚀" if result.tps >= 500 else "⚡" if result.tps >= 200 else "🐢"

        tps_table.add_row(
            "TPS (Tokens/Second)",
            f"[{tps_color}]{result.tps:,.1f} {tps_emoji}[/{tps_color}]"
        )
        tps_table.add_row("Requests/Second", f"{result.requests_per_second:.1f}")

        self.console.print(tps_table)

        # Latency metrics
        latency_table = Table(title="Latency")
        latency_table.add_column("Percentile", style="cyan")
        latency_table.add_column("Latency (ms)", style="magenta")

        latency_table.add_row("Average", f"{result.avg_latency_ms:.1f}ms")
        latency_table.add_row("P50 (Median)", f"{result.p50_latency_ms:.1f}ms")
        latency_table.add_row("P99", f"{result.p99_latency_ms:.1f}ms")

        self.console.print(latency_table)

        # Performance verdict
        rprint("\n" + "-" * 60)
        if result.tps >= 500:
            rprint(f"[bold green]Good throughput: {result.tps:,.0f} TPS[/bold green]")
        elif result.tps >= 200:
            rprint(f"[bold yellow]Moderate throughput: {result.tps:,.0f} TPS[/bold yellow]")
        else:
            rprint(f"[bold red]Low throughput: {result.tps:,.0f} TPS[/bold red]")
        print("-" * 60)


# ============================================================
# Benchmark Logic
# ============================================================

async def run_benchmark(
    config: BenchmarkConfig,
    progress: Progress,
) -> tuple[BenchmarkResult, float]:
    """Run the benchmark with concurrent requests.

    Returns:
        Tuple of (result, total_time_seconds) where time is measured
        from first request start to last request completion.
    """
    result = BenchmarkResult(total_requests=config.num_requests)

    async with RequestHandler(config) as handler:
        # --- Warmup requests ---
        # These warm up CUDA graphs, memory pools, and scheduler
        # Do NOT include in timing or results
        rprint("[dim]Warming up (3 requests)...[/dim]")
        for _ in range(3):
            await handler.make_request(config.prompts[0])

        # --- Timed benchmark requests ---
        task = progress.add_task("[cyan]Processing requests...", total=config.num_requests)

        semaphore = asyncio.Semaphore(config.concurrency)
        first_request_start: float | None = None
        last_request_end: float | None = None

        async def bounded_request(prompt: str) -> None:
            nonlocal first_request_start, last_request_end

            async with semaphore:
                # Record first request start time
                if first_request_start is None:
                    first_request_start = time.perf_counter()

                request_start = time.perf_counter()
                request_result = await handler.make_request(prompt)
                request_end = time.perf_counter()

                # Record last request end time
                last_request_end = request_end

                # Record latency (time from our send to receive)
                result.latencies_ms.append(request_result.latency_ms)
                result.total_tokens += request_result.tokens

                if request_result.success:
                    result.successful_requests += 1
                else:
                    result.failed_requests += 1

                progress.advance(task)

        # Create all tasks
        tasks = []
        for i in range(config.num_requests):
            prompt = config.prompts[i % len(config.prompts)]
            tasks.append(bounded_request(prompt))

        # Run with concurrency limit
        await asyncio.gather(*tasks, return_exceptions=True)

    # Calculate total time from first request to last response
    if first_request_start is not None and last_request_end is not None:
        total_time = last_request_end - first_request_start
    else:
        total_time = 0.0

    return result, total_time


# ============================================================
# Main Entry Point
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-4B vLLM deployment on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        type=str,
        help="API endpoint URL (or set MODAL_ENDPOINT in .env)",
    )
    parser.add_argument(
        "--requests",
        "-r",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help=f"Number of requests (default: {DEFAULT_NUM_REQUESTS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per request (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Measure cold start time before benchmark",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        help="Comma-separated prompts (or use default prompts)",
    )
    return parser.parse_args()


def main() -> int:
    """Main benchmark entry point."""
    console = Console()

    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_args()
    config = BenchmarkConfig.from_args(args)

    # Get endpoint from environment if not provided
    if not config.endpoint:
        config.endpoint = os.environ.get("MODAL_ENDPOINT", "")

    # Validate configuration
    if not config.validate():
        rprint("\n[cyan]Example:[/cyan]")
        rprint("  uv run python benchmark.py --endpoint https://your-app.modal.run/v1/chat/completions")
        rprint("  echo 'MODAL_ENDPOINT=https://...' > .env")
        return 1

    # Print header
    formatter = ResultFormatter(console)
    formatter.print_header(config)

    result = BenchmarkResult(total_requests=config.num_requests)

    # Measure cold start if requested
    if config.measure_cold_start:
        cold_start_ms = asyncio.run(measure_cold_start(config.endpoint))
        result.cold_start_ms = cold_start_ms
        if cold_start_ms < 0:
            rprint("[red]Server not available. Exiting.[/red]")
            return 1
        formatter.print_cold_start(cold_start_ms)

    # Create progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("({task.percentage:.0f}%)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # Run benchmark and get timing from first to last request
        result, result.total_time_seconds = asyncio.run(run_benchmark(config, progress))

    # Print results
    formatter.print_results(result, config.num_requests)

    # Save to JSON
    output_path = Path(config.output_file)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    rprint(f"\n[dim]Results saved to: {output_path}[/dim]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
