#!/usr/bin/env bash
# demo-phi-4-mini-instruct-fp8.sh — Serve the Flox-packaged FP8 model.
#
# The Phi-4-mini-instruct-FP8-TORCHAO model is installed as a Flox
# package and resolved via the "flox" source in vllm-resolve-model.
# No download required — the model is available immediately after
# flox activate.
#
# Usage:
#   flox activate -- ./examples/demo-phi-4-mini-instruct-fp8.sh
#
# Prerequisites:
#   - flox activate (provides vllm, vllm-serve, and the FP8 model package)
#   - GPU with ~5 GB free VRAM (Phi-4-mini FP8 is ~3.8B params, quantized)

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_MODEL="Phi-4-mini-instruct-FP8-TORCHAO"
export VLLM_MODEL_ORG="microsoft"
export VLLM_MODEL_SOURCES="flox"
export VLLM_SERVED_MODEL_NAME="Phi-4-mini-instruct"
export VLLM_PORT="${VLLM_PORT:-8000}"

echo "=== Demo: Flox-packaged FP8 model (${VLLM_MODEL_ORG}/${VLLM_MODEL}) ==="
echo ""

vllm-preflight && vllm-resolve-model && vllm-serve &
SERVE_PID=$!

MODEL_NAME="$VLLM_SERVED_MODEL_NAME"
USE_CHAT=1
export SERVE_PID MODEL_NAME USE_CHAT
source "${SCRIPT_DIR}/_test-server.sh"
