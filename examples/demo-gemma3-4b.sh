#!/usr/bin/env bash
# demo-gemma3-4b.sh — Serve Google Gemma 3 4B IT from the HuggingFace cache.
#
# Uses the "hf-cache" source in vllm-resolve-model to find models in
# ~/.cache/huggingface/hub/. Requires ~8 GB VRAM.
#
# Usage:
#   flox activate -- ./examples/demo-gemma3-4b.sh
#
# Prerequisites:
#   - flox activate (provides vllm, vllm-serve)
#   - Model downloaded in ~/.cache/huggingface/hub/
#   - GPU with ~8 GB free VRAM

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_MODEL="gemma-3-4b-it"
export VLLM_MODEL_ORG="google"
export VLLM_MODEL_SOURCES="hf-cache"
export VLLM_MODELS_DIR="$HOME/.cache/huggingface"
export VLLM_SERVED_MODEL_NAME="$VLLM_MODEL"
export VLLM_PORT="${VLLM_PORT:-8000}"

echo "=== Demo: HF cache model (${VLLM_MODEL_ORG}/${VLLM_MODEL}) ==="
echo "    VLLM_MODELS_DIR=${VLLM_MODELS_DIR}"
echo ""

vllm-preflight && vllm-resolve-model && vllm-serve &
SERVE_PID=$!

MODEL_NAME="$VLLM_MODEL"
USE_CHAT=1
export SERVE_PID MODEL_NAME USE_CHAT
source "${SCRIPT_DIR}/_test-server.sh"
