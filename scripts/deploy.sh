#!/bin/bash
# deploy.sh - Deploy Qwen3-4B vLLM service to Modal
# Usage: ./deploy.sh [--profile PROFILE]

set -e

PROFILE="${1:-default}"
APP_NAME="qwen3-4b-vllm"

echo "=============================================="
echo "Deploying Qwen3-4B vLLM to Modal (A10G)"
echo "=============================================="
echo ""
echo "Cost reminder: A10G is ~\$0.40/hour"
echo "Budget: \$5 credit available"
echo ""
echo "After deployment:"
echo "  1. Note your endpoint URL"
echo "  2. Run: ./benchmark.sh \$ENDPOINT"
echo "  3. Run: ./teardown.sh when done"
echo ""
echo "=============================================="

# Deploy the app
echo "Deploying to Modal..."
modal deploy src/modal_app.py --name "$APP_NAME"

echo ""
echo "=============================================="
echo "Deployment complete!"
echo "=============================================="
echo ""
echo "Your service endpoint will be:"
echo "  https://\$APP_NAME.modal.run/v1/chat/completions"
echo ""
echo "Wait ~2-5 minutes for model to load, then run benchmark."
