#!/bin/bash
# benchmark.sh - Run benchmark against deployed service
# Usage: ./benchmark.sh [ENDPOINT] [--cold-start]
#
# Uses a separate lightweight venv to avoid GPU dependencies.
# The benchmark is a client that runs locally, not on the GPU server.

set -e

# Load endpoint from .env if available
if [ -f .env ]; then
    source .env
fi

ENDPOINT="${1:-$MODAL_ENDPOINT}"
COLD_START="$2"

if [ -z "$ENDPOINT" ]; then
    echo "Usage: ./benchmark.sh [ENDPOINT] [--cold-start]"
    echo ""
    echo "Arguments:"
    echo "  ENDPOINT     Service URL (e.g., https://your-app.modal.run)"
    echo "              Will use MODAL_ENDPOINT from .env if not provided"
    echo "  --cold-start (optional) Measure cold start time"
    exit 1
fi

# Create lightweight venv for benchmark client (no GPU deps)
BENCH_VENV=".benchvenv"

if [ ! -d "$BENCH_VENV" ]; then
    echo "Creating lightweight benchmark environment..."
    python3 -m venv "$BENCH_VENV"
fi

# Activate and install minimal deps
source "$BENCH_VENV/bin/activate"
pip install --quiet httpx rich python-dotenv

echo "=============================================="
echo "Running benchmark..."
echo "=============================================="

# Build command
CMD="python src/benchmark.py --endpoint $ENDPOINT --output benchmark_results.json"

if [ "$COLD_START" = "--cold-start" ]; then
    CMD="$CMD --cold-start"
fi

echo "Running: $CMD"
echo ""

eval "$CMD"

deactivate 2>/dev/null || true

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "=============================================="
echo ""
echo "Results saved to: benchmark_results.json"
echo ""
echo "Next: Update README.md with your results, then run ./teardown.sh"
