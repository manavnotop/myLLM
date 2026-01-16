"""
Modal deployment for Qwen3-4B-Instruct using vLLM's built-in OpenAI-compatible server.

This uses vLLM's native server instead of custom FastAPI + Modal RPC for:
1. Industry-standard OpenAI API (familiar to interviewers)
2. No Modal RPC overhead - direct HTTP serving
3. Fewer lines of code
4. Built-in metrics and health endpoints

GPU: A10G (24GB) - cost-optimized alternative to A100
"""

import modal
from modal import Image, web_server

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# vLLM server args optimized for A10G (24GB)
# Note: A10G doesn't support FP8, so we use FP16
VLLM_ARGS = [
    "--host", "0.0.0.0",
    "--port", "8000",
    "--tensor-parallel-size", "1",
    "--gpu-memory-utilization", "0.85",  # Safety margin to avoid OOM
    "--max-model-len", "8192",           # Reduced for A10G memory
    "--trust-remote-code",
    "--disable-log-requests",
]

# Container image with GPU support and vLLM
IMAGE = (
    Image.from_registry(
        "nvidia/cuda:12.3.2-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install([
        "vllm>=0.6.6",
        "modal>=0.70",
        "httpx>=0.28.0",
    ])
)

# ============================================================
# Modal App Definition
# ============================================================

app = modal.App(
    name="qwen3-4b-vllm",
    image=IMAGE,
)

# ============================================================
# Web Server Entry Point
# ============================================================

@app.function(
    gpu="A10G",  # 24GB GPU, ~$0.40/hr
    concurrency_limit=256,
    timeout=1200,
    allow_concurrent_inputs=256,
)
@web_server(port=8000, startup_timeout=1200)
def serve():
    """
    Start vLLM's built-in OpenAI-compatible server.

    This runs as a Modal web server with:
    - /v1/chat/completions - OpenAI-compatible chat endpoint
    - /v1/models - List models
    - /health - Simple health check (may not be reliable)
    """
    import subprocess
    import sys

    cmd = [
        "vllm", "serve",
        MODEL_NAME,
        *VLLM_ARGS,
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.Popen(cmd)


if __name__ == "__main__":
    modal.run()
