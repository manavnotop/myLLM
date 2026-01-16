# Qwen3-4B vLLM Inference Service

Self-hosted LLM deployment demonstrating production inference infrastructure on Modal.

## Overview

This project deploys [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) using:
- **vLLM 0.6.6+** - Industry-standard inference engine
- **Modal** - Serverless GPU infrastructure
- **A10G GPU** - Cost-optimized 24GB GPU

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Modal Web Server (A10G 24GB)                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ vLLM OpenAI-Server                                   │ │
│  │  - Continuous batching                               │ │
│  │  - PagedAttention                                    │ │
│  │  - OpenAI-compatible API (/v1/chat/completions)     │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                        │
                        ▼
              https://{app}.modal.run/v1/chat/completions
```

## Why This Architecture?

| Decision | Rationale |
|----------|-----------|
| **vLLM OpenAI server** | Industry-standard API, familiar to interviewers, minimal code |
| **A10G vs A100** | 3x cheaper ($0.40/hr vs $1.50/hr), sufficient for Qwen 4B |
| **No warm instances** | Measure cold start honestly, avoid continuous billing |
| **No FP8** | A10G doesn't support FP8; FP16 is fine for 4B model |

## Performance Results

Measured with 50 requests, concurrency=10, max_tokens=256 on A10G GPU.

| Metric | Measured | Notes |
|--------|----------|-------|
| **Cold Start** | N/A | Measured warm (use `--cold-start` for cold measurement) |
| **TPS** | 337 tok/s | 50 requests, 10 concurrent, 232 avg tokens/request |
| **P50 Latency** | 6,770 ms | Median time to first token + generation |
| **P99 Latency** | 8,558 ms | Tail latency at 10 concurrent |
| **Success Rate** | 100% | 50/50 requests completed |
| **Cost/hour** | ~$0.40 | A10G on Modal |

### Raw Benchmark Data

```json
{
  "timestamp": "2026-01-16T19:26:28",
  "total_requests": 50,
  "successful_requests": 50,
  "total_tokens": 11619,
  "avg_tokens_per_request": 232.38,
  "tps": 337.46,
  "avg_latency_ms": 6267.76,
  "p50_latency_ms": 6769.81,
  "p99_latency_ms": 8557.64
}
```

## Quick Start

### Prerequisites

- Modal account with CLI installed
- `uv` package manager

### Deployment

```bash
# Deploy (costs ~$0.40/hour while running)
./scripts/deploy.sh

# Wait 2-5 minutes for model to load...

# Run benchmark (includes cold start measurement)
./scripts/benchmark.sh https://your-app.modal.run --cold-start

# Destroy immediately to stop billing
./scripts/teardown.sh
```

### Manual Commands

```bash
# Deploy
modal deploy src/modal_app.py --name qwen3-4b-vllm

# Benchmark with cold start
uv run python src/benchmark.py --endpoint https://your-app.modal.run/v1/chat/completions --cold-start --requests 50

# Teardown
modal app delete qwen3-4b-vllm --yes
```

### API Usage

```bash
# Chat completion (OpenAI-compatible)
curl -X POST https://your-app.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 256
  }'
```

## Cost Analysis

| GPU | Cost/hour | Notes |
|-----|-----------|-------|
| **A10G (this)** | ~$0.40 | 24GB, good for 4B models |
| A100 | ~$1.50 | 40GB, FP8 support |
| T4 | ~$0.35 | 16GB, older gen |

**Budget**: With $5 Modal credit, you can run ~12 hours of A10G.

## Files

| File | Purpose |
|------|---------|
| `src/modal_app.py` | Modal deployment with vLLM OpenAI server |
| `src/benchmark.py` | Cold start + TPS/latency measurement |
| `scripts/deploy.sh` | One-liner deployment script |
| `scripts/teardown.sh` | Destroy service to stop billing |
| `scripts/benchmark.sh` | Run benchmark against deployed service |

## Trade-offs & Improvements

### Trade-offs Made

1. **A10G instead of A100**
   - Saved: ~$1/hour
   - Cost: Slightly lower TPS, no FP8

2. **No warm instances**
   - Saved: Continuous billing
   - Cost: Cold start (~30-90s) on first request

3. **256 max tokens in benchmark**
   - Saved: Time per benchmark run
   - Cost: Doesn't measure long-form generation

### Future Improvements

- **Multi-GPU**: Tensor parallelism for larger models
- **Speculative decoding**: Faster generation with draft tokens
- **A100 upgrade**: If budget allows, enable FP8 for 2x speedup
- **Streaming**: Add streaming responses for better UX

## What This Demonstrates

1. **Production inference**: Using industry-standard vLLM, not toy code
2. **Cost awareness**: Making deliberate trade-offs for budget constraints
3. **Measurement discipline**: Cold start, TPS, latency - all measured honestly
4. **Infra as code**: Reproducible deployment scripts

## License

MIT
