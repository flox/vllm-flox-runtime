#!/usr/bin/env bash
# demo-flox-model.sh — Serve the model bundled in the flox manifest.
#
# The Phi-4-mini-instruct-FP8-TORCHAO model is installed as a flox package
# (phi-4-mini-instruct-fp8-hf) and resolved via the "flox" source in
# vllm-resolve-model. No network access or local model download required.
#
# Usage:
#   flox activate -- ./examples/demo-flox-model.sh
#
# Prerequisites:
#   - flox activate (provides vllm, vllm-serve, and the bundled model)
#   - GPU with ~4 GB free VRAM (Phi-4-mini FP8 is ~3.8B params quantized)

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_MODEL="Phi-4-mini-instruct-FP8-TORCHAO"
export VLLM_MODEL_ORG="microsoft"
export VLLM_MODEL_SOURCES="flox"
export VLLM_SERVED_MODEL_NAME="$VLLM_MODEL"
export VLLM_PORT="${VLLM_PORT:-8000}"

echo "=== Demo: flox-bundled model (${VLLM_MODEL_ORG}/${VLLM_MODEL}) ==="
echo ""

vllm-preflight && vllm-resolve-model && vllm-serve &
SERVE_PID=$!

MODEL_NAME="$VLLM_MODEL"
USE_CHAT=1
export SERVE_PID MODEL_NAME USE_CHAT
source "${SCRIPT_DIR}/_test-server.sh"
