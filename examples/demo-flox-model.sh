#!/usr/bin/env bash
# demo-flox-model.sh — Serve the locally downloaded model.
#
# The Phi-4-mini-instruct model is downloaded from GitHub Releases
# on first activation and resolved via the "local" source in
# vllm-resolve-model. No HuggingFace account or token required.
#
# Usage:
#   flox activate -- ./examples/demo-flox-model.sh
#
# Prerequisites:
#   - flox activate (provides vllm, vllm-serve, and the downloaded model)
#   - GPU with ~8 GB free VRAM (Phi-4-mini bfloat16 is ~3.8B params)

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_MODEL="Phi-4-mini-instruct"
export VLLM_MODEL_ORG="microsoft"
export VLLM_MODEL_SOURCES="local"
export VLLM_SERVED_MODEL_NAME="$VLLM_MODEL"
export VLLM_PORT="${VLLM_PORT:-8000}"

echo "=== Demo: local model (${VLLM_MODEL_ORG}/${VLLM_MODEL}) ==="
echo ""

vllm-preflight && vllm-resolve-model && vllm-serve &
SERVE_PID=$!

MODEL_NAME="$VLLM_MODEL"
USE_CHAT=1
export SERVE_PID MODEL_NAME USE_CHAT
source "${SCRIPT_DIR}/_test-server.sh"
