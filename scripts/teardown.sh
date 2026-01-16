#!/bin/bash
# teardown.sh - Destroy Modal deployment to stop billing
# Usage: ./teardown.sh [--profile PROFILE]

set -e

PROFILE="${1:-default}"
APP_NAME="qwen3-4b-vllm"

echo "=============================================="
echo "Destroying Modal deployment..."
echo "=============================================="
echo ""
echo "This will stop all billing immediately."
echo ""

# Stop the app
modal app stop "$APP_NAME"

echo ""
echo "=============================================="
echo "Deployment destroyed."
echo "=============================================="
echo ""
echo "Billing has stopped. Run ./deploy.sh to redeploy."
