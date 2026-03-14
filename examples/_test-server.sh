#!/usr/bin/env bash
# _test-server.sh — shared helper: wait for server, run smoke tests, shut down.
# Sourced by demo scripts after they launch vllm-serve in the background.
#
# Expects:
#   SERVE_PID    — PID of the background vllm-serve process
#   MODEL_NAME   — served-model-name for API requests
#   VLLM_PORT    — port (default 8000)
#   VLLM_API_KEY — bearer token (default sk-vllm-local-dev)
#
# Optional:
#   USE_CHAT=1   — also test /v1/chat/completions (requires instruct model)

set -Eeuo pipefail

: "${SERVE_PID:?SERVE_PID must be set before sourcing _test-server.sh}"
: "${MODEL_NAME:?MODEL_NAME must be set before sourcing _test-server.sh}"

_port="${VLLM_PORT:-8000}"
_key="${VLLM_API_KEY:-sk-vllm-local-dev}"
_base="http://127.0.0.1:${_port}"
_ok=0

cleanup() {
  if kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "--- Stopping server (PID $SERVE_PID) ---"
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "--- Waiting for server (up to 180s) ---"
for i in $(seq 1 180); do
  if curl -sf "${_base}/health" >/dev/null 2>&1; then
    echo "Server healthy after ${i}s"
    _ok=1
    break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "FAIL: server process died before becoming healthy"
    exit 1
  fi
  sleep 1
done

if (( ! _ok )); then
  echo "FAIL: server did not become healthy within 180s"
  exit 1
fi

echo ""
echo "--- /health ---"
curl -sf "${_base}/health" && echo " OK"

echo ""
echo "--- /v1/models ---"
curl -sf "${_base}/v1/models" \
  -H "Authorization: Bearer ${_key}" | python3 -m json.tool

echo ""
echo "--- /v1/completions ---"
curl -sf "${_base}/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${_key}" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 16
  }" | python3 -m json.tool

if [[ "${USE_CHAT:-0}" == "1" ]]; then
  echo ""
  echo "--- /v1/chat/completions ---"
  curl -sf "${_base}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${_key}" \
    -d "{
      \"model\": \"${MODEL_NAME}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer with just the number.\"}],
      \"max_tokens\": 16
    }" | python3 -m json.tool
fi

echo ""
echo "--- PASS ---"
