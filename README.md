# vLLM Runtime

Production vLLM inference server as a Flox environment. Installs `flox/vllm-flox-runtime` (model provisioning and serving scripts) and `flox-cuda/python3Packages.vllm` (vLLM + CUDA + Python) from the Flox catalog.

- **vLLM**: 0.15.1
- **CUDA**: requires NVIDIA driver with CUDA support
- **Platform**: Linux only (`x86_64-linux`)

## Quick start

```bash
# Activate and start the vLLM service
flox activate --start-services

# Override the model at activation time
VLLM_MODEL=DeepSeek-R1-Distill-Qwen-7B \
VLLM_MODEL_ORG=deepseek-ai \
  flox activate --start-services
```

### Verify it's running

```bash
# Health check (no auth required)
curl http://127.0.0.1:8000/health

# List loaded models
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer sk-vllm-local-dev"

# Chat completion
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm-local-dev" \
  -d '{
    "model": "Phi-3.5-mini-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

## Examples

The `examples/` directory contains self-contained demo scripts that start a vLLM server, run smoke tests (health, models, completions, chat), and shut down cleanly.

### Default model (Phi-3.5-mini-instruct-AWQ)

The default model is `microsoft/Phi-3.5-mini-instruct-AWQ` (~2.2 GB, AWQ 4-bit quantization). It is installed as a Flox package via Nix store-path and resolved via the `flox` source. No download required — the model is available immediately after activation. AWQ 4-bit quantization works on all CUDA GPUs including Tesla T4 (sm75).

### Customizing

Override the default port:

```bash
VLLM_PORT=8800 flox activate --start-services
```

## Architecture

The service command chains three scripts in a pipeline:

```
vllm-preflight && vllm-resolve-model && vllm-serve
```

```
┌──────────────────────────────────────────────────────┐
│  Consuming Environment (.flox/env/manifest.toml)     │
│                                                      │
│  [install]                                           │
│    flox/vllm-flox-runtime       # 3-script pipeline  │
│    flox/vllm-python312-cuda*    # vLLM + CUDA        │
│    (optional) flox/vllm-flox-monitoring              │
│                                                      │
│  [services]                                          │
│    vllm → vllm-preflight                             │
│           && vllm-resolve-model                      │
│           && vllm-serve                              │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  vllm-preflight                                 │ │
│  │    Port reclaim ← /proc/net/tcp + /proc/<pid>/  │ │
│  │    GPU health   ← NVML → nvidia-smi → skip      │ │
│  ├─────────────────────────────────────────────────┤ │
│  │  vllm-resolve-model                             │ │
│  │    Sources: flox → local → hf-cache → r2 → hub │ │
│  │    Output: per-model .env file (mode 600)       │ │
│  ├─────────────────────────────────────────────────┤ │
│  │  vllm-serve                                     │ │
│  │    Loads .env → validates args → exec vllm serve│ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

1. **vllm-preflight** — Reclaims the port if occupied by a stale vLLM process, checks GPU health via NVML or nvidia-smi, optionally executes a downstream command.
2. **vllm-resolve-model** — Provisions the model from configured sources with locking and atomic swaps, validates the model directory (config, tokenizer, weight shards), writes a per-model env file.
3. **vllm-serve** — Loads the env file (safe or trusted mode), validates all required vars, builds the `vllm serve` argv from env vars + `config.yaml`, and `exec`s.

Scripts are provided by the `flox/vllm-flox-runtime` package (~1,700 lines of hardened Bash) and available on `PATH` after activation.

## API reference

The server exposes an OpenAI-compatible API. All authenticated endpoints require the `Authorization: Bearer <VLLM_API_KEY>` header.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | `GET` | No | Health check. Returns `200 OK` when ready |
| `/v1/models` | `GET` | Yes | List loaded models |
| `/v1/chat/completions` | `POST` | Yes | Chat completions (streaming supported) |
| `/v1/completions` | `POST` | Yes | Text completions (streaming supported) |
| `/metrics` | `GET` | No | Prometheus metrics |

### Chat completion

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm-local-dev" \
  -d '{
    "model": "Phi-3.5-mini-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain TCP in one paragraph."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Streaming

```bash
curl --no-buffer http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm-local-dev" \
  -d '{
    "model": "Phi-3.5-mini-instruct",
    "messages": [{"role": "user", "content": "Write a haiku about CUDA."}],
    "max_tokens": 64,
    "stream": true
  }'
```

### Text completion

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm-local-dev" \
  -d '{
    "model": "Phi-3.5-mini-instruct",
    "prompt": "The capital of France is",
    "max_tokens": 32
  }'
```

## Configuration reference

Settings are split between a static config file and runtime environment variables.

### Static settings (`vllm-config.yaml`)

A default config is bundled in the `vllm-flox-runtime` package and auto-copied to `$FLOX_ENV_CACHE/vllm-config.yaml` on first run. Edit that copy to customize. These settings are read by `vllm-serve` and passed directly to `vllm serve` via `--config`. `host` and `port` are overridden by the `VLLM_HOST`/`VLLM_PORT` env vars (passed as CLI args, which take precedence over config file values).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8000` | HTTP listen port |
| `dtype` | `float16` | Weight data type. Use `float16` for AWQ-quantized models; `auto` selects BF16 for BF16 models, FP16 for FP16/FP32 models |
| `gpu-memory-utilization` | `0.85` | Per-GPU VRAM fraction for KV cache. Reduce if you see OOM during prefill. Increase for cards with more headroom (e.g., 0.92 for 24 GB, 0.95 for 48 GB+) |
| `quantization` | `awq` | Quantization method. Set to match the model's quantization (e.g., `awq`, `gptq`). Remove for unquantized models |
| `disable-log-requests` | `true` | Suppress per-request logging |
| `uvicorn-log-level` | `warning` | Uvicorn server log level |

### Runtime environment variables (on-activate hook)

All vars use `${VAR:-default}` in the on-activate hook so they can be overridden at activation time:

```bash
VLLM_MAX_MODEL_LEN=16384 VLLM_KV_CACHE_DTYPE=fp8 flox activate --start-services
```

#### Model settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `Phi-3.5-mini-instruct-AWQ` | Model directory name. Must be a single safe path element (no `/`, `\`, `.`, `..`, or control characters) |
| `VLLM_MODEL_ORG` | `microsoft` | HuggingFace org. Used to derive the model ID as `$VLLM_MODEL_ORG/$VLLM_MODEL` when `VLLM_MODEL_ID` is not set |
| `VLLM_MODEL_SOURCES` | `flox,local,hf-cache,hf-hub` | Comma-separated source order for model provisioning. Available sources: `flox`, `local`, `hf-cache`, `r2`, `hf-hub` |
| `VLLM_MODELS_DIR` | `$FLOX_ENV_PROJECT/models` | Root directory for model storage and HF cache. Created automatically on activation |
| `VLLM_SERVED_MODEL_NAME` | `Phi-3.5-mini-instruct` | Model name returned in `/v1/models` responses and used in API requests |

#### Server settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST` | `0.0.0.0` | Server bind address |
| `VLLM_PORT` | `8000` | Server listen port. Must be 1-65535 |
| `VLLM_API_KEY` | `sk-vllm-local-dev` | Bearer token for API authentication |

#### Engine tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism. Must be > 0 |
| `VLLM_PIPELINE_PARALLEL_SIZE` | `1` | Number of GPUs for pipeline parallelism. Must be > 0 |
| `VLLM_PREFIX_CACHING` | `false` | Automatic prefix caching. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `VLLM_KV_CACHE_DTYPE` | `auto` | KV cache precision. `auto` matches model dtype; `fp8` halves KV cache memory at minor quality cost. Must not contain whitespace |
| `VLLM_MAX_MODEL_LEN` | `4096` | Max sequence length (input + output tokens). Must not exceed the model's native context length. Lower values reduce memory. Must be > 0 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `4096` | Chunked prefill budget. Increase for throughput at the cost of higher per-request latency. Must be > 0 |

#### Logging and metrics

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_LOGGING_LEVEL` | `WARNING` | vLLM Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `PROMETHEUS_MULTIPROC_DIR` | `$FLOX_ENV_CACHE/vllm-prometheus` | Directory for Prometheus client multiprocess metrics. Created automatically on activation |

## Model provisioning (`vllm-resolve-model`)

Searches configured sources in order, validates the model directory, and writes an env file that `vllm-serve` loads. The first source that produces a valid model wins.

### Source table

Sources are tried in the order specified by `VLLM_MODEL_SOURCES`. The script's internal default is `flox,local,hf-cache,r2,hf-hub`; the manifest sets `flox,local,hf-cache,hf-hub`.

| Source | What it checks | Skip condition | Resolution |
|--------|---------------|----------------|------------|
| `flox` | `$FLOX_ENV/share/models/hub/models--<slug>/snapshots/` | `FLOX_ENV` not set | Sets `HF_HOME` to the flox package model path |
| `local` | `$VLLM_MODELS_DIR/<VLLM_MODEL>/` | Directory missing or fails validation | Sets `VLLM_MODEL_PATH` to the local directory |
| `hf-cache` | `$VLLM_MODELS_DIR/hub/models--<slug>/snapshots/` | No usable snapshot found | Sets `HF_HOME` to `$VLLM_MODELS_DIR` |
| `r2` | Downloads from `s3://$R2_BUCKET/$R2_MODELS_PREFIX/$VLLM_MODEL/` | `aws` CLI missing, `R2_BUCKET`/`R2_MODELS_PREFIX` not set, or credentials fail | Stages to temp dir, validates, atomic-swaps into `$VLLM_MODELS_DIR/<VLLM_MODEL>/` |
| `hf-hub` | Downloads from HuggingFace Hub using `hf`/`huggingface-cli`/`python3` | No download tool found | Stages to temp dir, validates, atomic-swaps into `$VLLM_MODELS_DIR/<VLLM_MODEL>/` |

### Environment variables

**Required:**

| Variable | Description |
|----------|-------------|
| `VLLM_MODEL` | Model name (single safe path element) |
| `VLLM_MODELS_DIR` | Base directory for local models and HF cache |

**Optional:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL_ID` | Derived from `$VLLM_MODEL_ORG/$VLLM_MODEL` | Explicit HuggingFace model ID (`org/name`). When empty, derived from `VLLM_MODEL_ORG` (which must then be set) |
| `VLLM_MODEL_ORG` | _(none; manifest sets `microsoft`)_ | Org prefix for deriving model ID. Required when `VLLM_MODEL_ID` is empty |
| `VLLM_MODEL_SOURCES` | `flox,local,hf-cache,r2,hf-hub` | Comma-separated source order |
| `FLOX_ENV` | _(set by Flox)_ | Flox environment path. Required for `flox` source |
| `FLOX_ENV_CACHE` | _(set by Flox)_ | Cache directory for env files. Required when `VLLM_MODEL_ENV_FILE` is not set |
| `VLLM_MODEL_ENV_FILE` | `$FLOX_ENV_CACHE/vllm-model.<slug>.<hash>.env` | Override env file output path |
| `R2_BUCKET` | _(none)_ | Cloudflare R2 bucket name. Required for `r2` source |
| `R2_MODELS_PREFIX` | _(none)_ | R2 key prefix for models. Required for `r2` source |
| `R2_ENDPOINT_URL` | _(none)_ | AWS CLI endpoint URL for R2 |
| `VLLM_RESOLVE_LOCK_TIMEOUT` | `300` | Seconds to wait for the per-model lock |
| `VLLM_SKIP_TOKENIZER_CHECK` | `0` | Set to `1` to skip tokenizer asset validation |
| `VLLM_KEEP_LOGS` | `0` | Set to `1` to keep download logs even on success. Logs are always kept on failure |
| `HF_TOKEN` | _(none)_ | HuggingFace token for gated model access |

### Model validation

Every candidate model directory must pass three checks before it is accepted:

1. **`config.json`** — must exist at the directory root.
2. **Tokenizer assets** — at least one recognized tokenizer file must exist in `<dir>/`, `<dir>/tokenizer/`, or `<dir>/tokenizer_files/`. Recognized files:
   - `tokenizer.json`, `tokenizer.model`, `spiece.model`
   - `vocab.json` + `merges.txt`
   - `vocab.txt`
   - `tokenizer_config.json` + (`vocab.json` or `vocab.txt`)
   - Skip this check with `VLLM_SKIP_TOKENIZER_CHECK=1`.
3. **Weight shards** — determined by the presence of shard index files:
   - If `*.index.json` exists: all shard files referenced in `weight_map` must exist.
   - If no index but `-00001-of-NNNNN` pattern detected: all N shards must exist.
   - Otherwise: at least one weight-like file (`.safetensors`, `.bin`, `.pt`, `.pth`, `.gguf`) must exist.

### Env file output

Written atomically (mktemp + mv) to `$FLOX_ENV_CACHE/vllm-model.<slug>.<hash12>.env` with mode `600` (umask `077`). Contains:

```bash
# generated by vllm-resolve-model
export HF_HOME='/path/to/hf/home'           # when resolved via flox or hf-cache
export VLLM_MODEL='Llama-3.1-8B-Instruct'
export VLLM_MODEL_ID='meta-llama/Llama-3.1-8B-Instruct'
export VLLM_MODEL_PATH='/path/to/models/Llama-3.1-8B-Instruct'  # when resolved locally
export _VLLM_RESOLVED_MODEL='meta-llama/Llama-3.1-8B-Instruct'
export _VLLM_RESOLVED_VIA='hf-hub'
```

The `<slug>` is the model ID with unsafe characters mapped to `-`. The `<hash12>` is the first 12 hex chars of SHA-256 of the model ID, computed using whichever is available: `sha256sum`, `shasum`, `openssl`, or `python3`.

### Gated models

Gated models that require authentication need a HuggingFace token:

```bash
HF_TOKEN=hf_... flox activate --start-services
```

### Offline operation

Restrict sources to avoid network access:

```bash
VLLM_MODEL_SOURCES=local flox activate --start-services           # local only
VLLM_MODEL_SOURCES=local,hf-cache flox activate --start-services  # local + cached
```

### Locking and atomic swap

- **Per-model lock**: acquired before any source search. Uses `flock` if available, falls back to `mkdir`-based locking with stale PID detection. Timeout: `VLLM_RESOLVE_LOCK_TIMEOUT` seconds (default 300).
- **Atomic swap** (r2 and hf-hub only): downloads stage into a temp directory under `$VLLM_MODELS_DIR/.staging/`. After validation, the staged directory replaces the target via backup+rename. If interrupted, the next run restores the newest backup automatically.

## Pre-flight (`vllm-preflight`)

Pre-flight validation: reclaims the vLLM port if occupied, checks GPU health, and optionally executes a downstream command.

**Platform**: Linux only (requires `/proc`).

### Usage

```bash
vllm-preflight                        # checks only
vllm-preflight ./start.sh arg1 arg2   # checks, then exec command
vllm-preflight -- python -m vllm ...  # checks, then exec command (after --)
```

### Exit codes

Stable contract — these codes are safe to match on programmatically.

| Code | Meaning | When |
|------|---------|------|
| `0` | Success | Port free (or reclaimed), GPU OK, downstream command exec'd |
| `1` | Validation error | Bad env var value, GPU hard failure, bad config, `python3` not found |
| `2` | Port owned by non-vLLM process | A non-vLLM listener holds the port. Will not kill |
| `3` | Different UID | vLLM process on the port belongs to another user. Will not kill (unless `VLLM_ALLOW_OTHER_UID_KILL=1`) |
| `4` | Not attributable | Listener found but cannot map socket inodes to PIDs (permissions / hidepid) |
| `5` | Stop failed | Sent SIGTERM/SIGKILL but port is still listening after timeout |

In `--dry-run` mode (`VLLM_DRY_RUN=1`), exit codes are `0`/`2`/`3`/`4` only (never `5`, since nothing is killed).

### Environment variables

| Variable | Default | Validation | Description |
|----------|---------|------------|-------------|
| `VLLM_HOST` | `0.0.0.0` | IP/hostname | Bind address to check |
| `VLLM_PORT` | `8000` | Integer, 1-65535 | Port to check and reclaim |
| `VLLM_DRY_RUN` | `0` | `0` or `1` | Report what would happen without sending signals |
| `VLLM_SKIP_GPU_CHECK` | `0` | `0` or `1` | Skip all GPU checks |
| `VLLM_MIN_FREE_GPU_GB` | `4` | Numeric, >= 0 | Minimum free GPU memory (GiB). Hard-fails if `memory` in `VLLM_GPU_FAIL_ON` |
| `VLLM_MAX_GPU_TEMP_C` | `85` | Integer, >= 1 | Hard-fail if GPU temperature exceeds this (Celsius) |
| `VLLM_MAX_GPU_UTIL_PCT` | `95` | Integer, 0-100 | Hard-fail if GPU utilization exceeds this percentage |
| `VLLM_GPU_FAIL_ON` | `temperature` | Comma-separated | Conditions that trigger hard failure: `temperature`, `memory`, `utilization` |
| `VLLM_GPU_DEVICES` | _(unset)_ | CSV | GPU device indices/UUIDs to check. Falls back to `CUDA_VISIBLE_DEVICES` |
| `VLLM_ALLOW_OTHER_UID_KILL` | `0` | `0` or `1` | Allow killing vLLM processes owned by other UIDs |
| `VLLM_STOP_TIMEOUT` | `15` | Integer, >= 0 | Seconds for full stop cycle (SIGTERM → SIGKILL) |
| `VLLM_PROCESS_SIGNATURES` | _(unset)_ | Comma-separated | Additional cmdline signatures to identify as vLLM processes |
| `VLLM_PREFLIGHT_LOCKFILE` | `/tmp/vllm-preflight.{port}.lock` | File path | Lock file path. Port-keyed by default |
| `VLLM_PREFLIGHT_JSON` | `0` | `0` or `1` | JSON output on stdout. Incompatible with downstream command |
| `VLLM_PREFLIGHT_PROXY_CHILD` | `1` | `0` or `1` | Proxy mode: start child, wait for bind, forward signals. Set `0` for plain exec |
| `VLLM_START_BIND_TIMEOUT` | `60` | Numeric, > 0 | Max seconds to wait for downstream to bind the target port (proxy mode) |
| `VLLM_START_BIND_POLL` | `0.2` | Numeric, > 0 | Poll interval while waiting for downstream bind (proxy mode) |

`VLLM_PROCESS_SIGNATURES` example for unusual launchers:

```bash
VLLM_PROCESS_SIGNATURES='my.custom.launcher,another_entrypoint'
```

### Port reclaim behavior

1. Parses `/proc/net/tcp` and `/proc/net/tcp6` for LISTEN-state sockets matching the configured host and port (including wildcard `0.0.0.0`/`::` catchall).
2. Maps socket inodes to PIDs via `/proc/<pid>/fd/` symlink scanning.
3. Reads `/proc/<pid>/cmdline` and `/proc/<pid>/exe` to classify each listener as vLLM or non-vLLM:
   - **Built-in signatures**: `vllm.entrypoints.openai.api_server`, `vllm.entrypoints.api_server`, `vllm.entrypoints.openai.run_server`, `vllm serve`, `vllm.entrypoints`.
   - **Custom signatures**: set `VLLM_PROCESS_SIGNATURES` for unusual launchers.
4. **Non-vLLM listener** → exit 2 (refuses to kill).
5. **Different UID** → exit 3 (unless `VLLM_ALLOW_OTHER_UID_KILL=1`).
6. **Unmappable inodes** → exit 4 (e.g., hidepid restrictions).
7. **Own vLLM** → builds process tree via `/proc/<pid>/stat` parent chain, sends SIGTERM to all descendants (post-order), waits up to `VLLM_STOP_TIMEOUT` seconds (default 15), then SIGKILL any survivors.
8. Polls until port is free or `VLLM_STOP_TIMEOUT` expires. If still listening → exit 5.

### GPU health check

Runs after port reclaim. Three-tier cascade:

1. **NVML** (preferred): ctypes probe of `libcuda.so.1` + `libnvidia-ml.so.1`. Per-GPU: name, memory, temperature, utilization, pstate, clock throttle reasons. Hard-fails if driver present but 0 devices.
2. **nvidia-smi** (fallback): same fields via CSV output.
3. **Neither available**: warning, continue.

Threshold checks:

- **Memory**: hard-fails if free < `VLLM_MIN_FREE_GPU_GB` and `memory` in `VLLM_GPU_FAIL_ON`.
- **Temperature**: hard-fails if > `VLLM_MAX_GPU_TEMP_C` and `temperature` in `VLLM_GPU_FAIL_ON` (default: yes).
- **Utilization**: hard-fails if > `VLLM_MAX_GPU_UTIL_PCT` and `utilization` in `VLLM_GPU_FAIL_ON`.

### JSON output mode

When `VLLM_PREFLIGHT_JSON=1`, a single JSON object is printed to stdout. Human-readable logs still go to stderr. Incompatible with downstream command execution.

Examples:

```json
{"ok":true,"host":"0.0.0.0","port":8000,"port_action":"noop","reclaimed_roots":[],"blocked_roots":[],"could_scan":true,"gpus":[],"gpu_check_source":"skipped"}
{"ok":true,"host":"0.0.0.0","port":8000,"port_action":"stopped","reclaimed_roots":[12345],"blocked_roots":[],"could_scan":true,"gpus":[...],"gpu_check_source":"nvml"}
```

### Locking

Prevents two concurrent preflight runs from racing on the same port:

- **flock** (preferred): opens `$VLLM_PREFLIGHT_LOCKFILE` with `flock -n`. Validates the lockfile is not a symlink and is a regular file.
- **mkdir fallback**: creates `$VLLM_PREFLIGHT_LOCKFILE.d/` atomically. Detects stale locks via PID file.

## Serving (`vllm-serve`)

Loads the resolved model env file and executes `vllm serve` with validated arguments.

### Usage

```bash
vllm-serve                           # standard launch
vllm-serve --print-cmd               # print the vllm serve argv to stderr, then exec
vllm-serve --dry-run                 # print the argv to stderr and exit 0 (no exec)
vllm-serve -h                        # show help
vllm-serve -- --extra-flag val       # pass extra args through to vllm serve
```

### Required environment variables

**Always required:**

| Variable | Validation | Description |
|----------|------------|-------------|
| `FLOX_ENV_CACHE` | Must be a directory | Cache directory (for default `vllm-config.yaml`). Not required if `VLLM_CONFIG_FILE` is set |
| `VLLM_HOST` | Non-empty | Server bind address |
| `VLLM_PORT` | Positive integer | Server listen port |
| `VLLM_TENSOR_PARALLEL_SIZE` | Positive integer | Tensor parallelism GPU count |
| `VLLM_PIPELINE_PARALLEL_SIZE` | Positive integer | Pipeline parallelism GPU count |
| `VLLM_KV_CACHE_DTYPE` | Non-empty, no whitespace | KV cache dtype (e.g., `auto`, `fp8`) |
| `VLLM_MAX_MODEL_LEN` | Positive integer | Max sequence length |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | Positive integer | Chunked prefill budget |
| `VLLM_SERVED_MODEL_NAME` | Non-empty | Model name for API responses |

**Required when `VLLM_MODEL_ENV_FILE` is not set** (the standard case):

| Variable | Description |
|----------|-------------|
| `FLOX_ENV_CACHE` | Cache directory. Must exist as a directory |
| `VLLM_MODEL_ID` | Full model ID (`org/model`), OR `VLLM_MODEL_ORG` + `VLLM_MODEL` must both be set |

### Optional environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL_ENV_FILE` | Derived from `FLOX_ENV_CACHE` + model ID | Explicit env file path. Bypasses the standard derivation |
| `VLLM_PREFIX_CACHING` | `false` | Enable automatic prefix caching. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |
| `VLLM_CONFIG_FILE` | `$FLOX_ENV_CACHE/vllm-config.yaml` | Override config file path. Default is auto-copied from the package on first run |
| `VLLM_ENV_FILE_TRUSTED` | `false` | Skip safe-mode env file parsing and `source` the file directly. Accepts `true`/`false`/`1`/`0`/`yes`/`no` |

### Env file loading

Two modes:

**Safe mode** (default, `VLLM_ENV_FILE_TRUSTED=false`): The env file is parsed by a Python script that enforces a restricted `.env` subset:
- Lines must be `KEY=VALUE` or `export KEY=VALUE`.
- Values may be single-quoted, double-quoted, or unquoted.
- Double-quoted values support `\\`, `\"`, `\n`, `\t` escapes.
- Trailing `# comments` are allowed after quoted values.
- No multiline values, no `${VAR}` interpolation, no command substitution.
- Generates sanitized `export KEY='value'` lines, then `source`s the sanitized output.
- Requires `python3` (or `python`) on PATH.

**Trusted mode** (`VLLM_ENV_FILE_TRUSTED=true`): The env file is `source`d directly as shell code. Only use this if you fully trust the env file contents.

The env file must define `_VLLM_RESOLVED_MODEL` or `vllm-serve` exits with an error.

### Command construction

`vllm-serve` builds the final argv as:

```bash
vllm serve <_VLLM_RESOLVED_MODEL> \
  --config <config_file> \
  --host <VLLM_HOST> \
  --port <VLLM_PORT> \
  --tensor-parallel-size <VLLM_TENSOR_PARALLEL_SIZE> \
  --pipeline-parallel-size <VLLM_PIPELINE_PARALLEL_SIZE> \
  --kv-cache-dtype <VLLM_KV_CACHE_DTYPE> \
  --max-model-len <VLLM_MAX_MODEL_LEN> \
  --max-num-batched-tokens <VLLM_MAX_NUM_BATCHED_TOKENS> \
  --served-model-name <VLLM_SERVED_MODEL_NAME> \
  [--enable-prefix-caching]    # when VLLM_PREFIX_CACHING is true/1/yes
  [extra args...]              # anything after -- on the vllm-serve command line
```

The env var to vLLM CLI flag mapping:

| Env var | CLI flag |
|---------|----------|
| `_VLLM_RESOLVED_MODEL` | positional (model argument) |
| `VLLM_CONFIG_FILE` or `$FLOX_ENV_CACHE/vllm-config.yaml` | `--config` |
| `VLLM_HOST` | `--host` |
| `VLLM_PORT` | `--port` |
| `VLLM_TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size` |
| `VLLM_PIPELINE_PARALLEL_SIZE` | `--pipeline-parallel-size` |
| `VLLM_KV_CACHE_DTYPE` | `--kv-cache-dtype` |
| `VLLM_MAX_MODEL_LEN` | `--max-model-len` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` |
| `VLLM_SERVED_MODEL_NAME` | `--served-model-name` |
| `VLLM_PREFIX_CACHING` | `--enable-prefix-caching` (when truthy) |

### Config file resolution

1. If `VLLM_CONFIG_FILE` is set, use that path.
2. Otherwise, use `$FLOX_ENV_CACHE/vllm-config.yaml`.
3. If that file doesn't exist, the default config bundled in the package (`<pkg-root>/share/vllm-flox-runtime/config.yaml`) is auto-copied there on first run.
4. The file must exist and be readable, or `vllm-serve` exits with an error.

## Multi-GPU

```bash
# 2-way tensor parallel (most common)
VLLM_TENSOR_PARALLEL_SIZE=2 flox activate --start-services

# 4-way pipeline parallel
VLLM_PIPELINE_PARALLEL_SIZE=4 flox activate --start-services

# 4-way hybrid: TP=2 x PP=2
VLLM_TENSOR_PARALLEL_SIZE=2 \
VLLM_PIPELINE_PARALLEL_SIZE=2 \
  flox activate --start-services
```

**TP** (tensor parallelism) shards weight matrices across GPUs — reduces per-GPU memory, best for latency. **PP** (pipeline parallelism) distributes layers sequentially across GPUs — useful when TP alone isn't enough. **TP x PP must equal your total GPU count.**

## Swapping models

```bash
# Override at activation time
VLLM_MODEL=Qwen2.5-7B-Instruct \
VLLM_MODEL_ORG=Qwen \
  flox activate --start-services

# Or edit the on-activate defaults in manifest.toml and restart
flox services restart vllm
```

## Service management

```bash
flox services status              # check service state
flox services logs vllm           # tail service logs
flox services logs vllm -f        # follow logs
flox services restart vllm        # restart the vLLM service
flox services stop                # stop all services
flox activate --start-services    # activate and start in one step
```

## Monitoring

Install `flox/vllm-flox-monitoring` alongside this environment to add Prometheus + Grafana:

```toml
# Add to [install]
prometheus.pkg-path = "prometheus"
grafana.pkg-path = "grafana"
vllm-flox-monitoring.pkg-path = "flox/vllm-flox-monitoring"

# Add to end of on-activate
#   . vllm-monitoring-init

# Add to [services]
# prometheus.command = "vllm-monitoring-prometheus"
# grafana.command = "vllm-monitoring-grafana"
```

Key override env vars: `PROMETHEUS_PORT` (default `9090`), `GF_SERVER_HTTP_PORT` (default `3000`).

```bash
curl http://127.0.0.1:9090/api/v1/targets   # Prometheus targets
curl http://127.0.0.1:3000/api/health        # Grafana health
```

Raw vLLM metrics are always available at `http://127.0.0.1:8000/metrics` without additional packages.

## Kubernetes deployment

Deploy vLLM to Kubernetes using the Flox "Imageless Kubernetes" (uncontained) pattern. The Flox containerd shim pulls the environment from FloxHub at pod startup, replacing the need for a container image.

### Prerequisites

- A Kubernetes cluster with the [Flox containerd shim](https://flox.dev/docs/tutorials/kubernetes/) installed on GPU nodes
- NVIDIA GPU operator or device plugin configured
- A StorageClass that supports `ReadWriteOnce` PVCs

### Deploy

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### What the manifests do

| File | Purpose |
|------|---------|
| `k8s/namespace.yaml` | Creates the `vllm` namespace |
| `k8s/pvc.yaml` | 50 Gi `ReadWriteOnce` volume for model storage at `/models` |
| `k8s/deployment.yaml` | Single-replica pod with Flox shim, GPU resources, health probes |
| `k8s/service.yaml` | ClusterIP service on port 8000 |

The deployment uses `runtimeClassName: flox` and `image: flox/empty:1.0.0` — the Flox shim intercepts pod creation, pulls `barstoolbluz/vllm-runtime` from FloxHub, activates the environment, then runs the entrypoint (`vllm-preflight && vllm-resolve-model && vllm-serve`).

### Storage

Model weights are stored on the PVC mounted at `/models`. The pod sets `VLLM_MODELS_DIR=/models` to override the local default (`$FLOX_ENV_PROJECT/models`). The default Phi-3.5-mini-instruct-AWQ model is included as a Flox package and resolved via the `flox` source — no download required at startup.

Set the `storageClassName` in `k8s/pvc.yaml` to match your cluster:

```yaml
storageClassName: gp3  # AWS EBS
storageClassName: standard-rwo  # GKE
storageClassName: managed-premium  # AKS
```

### Secrets

Create a Kubernetes Secret for API authentication and gated model access, then uncomment the `secretKeyRef` blocks in the deployment:

```bash
kubectl -n vllm create secret generic vllm-secrets \
  --from-literal=api-key='your-production-api-key' \
  --from-literal=hf-token='hf_...'
```

Without the secret, `VLLM_API_KEY` defaults to `sk-vllm-local-dev` from the on-activate hook.

### Customizing the model

Override the model via pod environment variables:

```yaml
env:
  - name: VLLM_MODEL
    value: "Qwen2.5-7B-Instruct"
  - name: VLLM_MODEL_ORG
    value: "Qwen"
```

For multi-GPU inference, set `VLLM_TENSOR_PARALLEL_SIZE` and request additional GPUs:

```yaml
env:
  - name: VLLM_TENSOR_PARALLEL_SIZE
    value: "2"
resources:
  limits:
    nvidia.com/gpu: 2
```

### Startup timing

The `startupProbe` allows 10 minutes (60 failures x 10s) for warm starts with a cached model on the PVC. For cold starts (first-time model download), increase the threshold:

```yaml
startupProbe:
  failureThreshold: 120  # 20 minutes for cold start
```

Liveness and readiness probes are gated behind the startup probe and will not kill slow-starting pods.

### Verifying the deployment

```bash
# Watch pod startup
kubectl -n vllm get pods -w

# Check logs
kubectl -n vllm logs -f deployment/vllm

# Health check (from within the cluster)
kubectl -n vllm run curl --rm -it --image=curlimages/curl -- \
  curl http://vllm:8000/health

# Port-forward for local access
kubectl -n vllm port-forward svc/vllm 8000:8000
curl http://localhost:8000/health
```

### Exposing externally

The service defaults to `ClusterIP`. For external access, change the type or add an Ingress:

```bash
# Quick LoadBalancer
kubectl -n vllm patch svc vllm -p '{"spec":{"type":"LoadBalancer"}}'

# Or use port-forward for development
kubectl -n vllm port-forward svc/vllm 8000:8000
```

## Troubleshooting

Common issues and their solutions. Exit codes refer to `vllm-preflight`.

### Port conflict (exit code 2)

`vllm-preflight` automatically reclaims the port from stale vLLM processes. If it exits with code 2, a non-vLLM process is using the port.

```bash
# Find what's on the port
ss -tlnp | grep :8000

# Either stop that process or change the port
VLLM_PORT=8001 flox activate --start-services
```

### Different UID (exit code 3)

Another user's vLLM process holds the port. Either ask them to stop it, or:

```bash
VLLM_ALLOW_OTHER_UID_KILL=1 flox activate --start-services
```

### Inode not attributable (exit code 4)

Your system restricts `/proc/<pid>/fd` visibility (hidepid mount option). Run as the owning user, relax hidepid, or free the port manually.

### Stop failed (exit code 5)

The port is still listening after SIGTERM + SIGKILL + timeout. Increase the timeout or investigate manually:

```bash
VLLM_STOP_TIMEOUT=30 flox activate --start-services
```

### GPU not detected

Verify with `nvidia-smi`. This environment requires NVIDIA driver 575+. To skip the GPU check (e.g., for CPU-only testing):

```bash
VLLM_SKIP_GPU_CHECK=1 flox activate --start-services
```

### Gated model 401

Gated models require a HuggingFace token:

```bash
HF_TOKEN=hf_... flox activate --start-services
```

Ensure you've accepted the model's license on the HuggingFace website.

### Out of memory (OOM)

Reduce memory pressure:

1. Lower `gpu-memory-utilization` in `.flox/cache/vllm-config.yaml` (e.g., `0.85`).
2. Reduce `VLLM_MAX_MODEL_LEN` (e.g., `2048`).
3. Use `VLLM_KV_CACHE_DTYPE=fp8` to halve KV cache memory.
4. Increase tensor parallelism to spread the model across GPUs.

### Missing tokenizer

Some models use non-standard tokenizer layouts. Skip the check:

```bash
VLLM_SKIP_TOKENIZER_CHECK=1 flox activate --start-services
```

### Stale lock

If a previous run was killed mid-operation, the lock file may be stale:

```bash
# For vllm-preflight
rm -f /tmp/vllm-preflight.*.lock

# For vllm-resolve-model (lockfile is next to the env file)
rm -f "$FLOX_ENV_CACHE"/vllm-model.*.lock
```

The mkdir-based fallback lock includes stale PID detection and self-cleans.

### Env file not found

`vllm-serve` cannot find the model env file. Run `vllm-resolve-model` first:

```bash
vllm-resolve-model && vllm-serve
```

Or specify the env file explicitly:

```bash
VLLM_MODEL_ENV_FILE=/path/to/env vllm-serve
```

### Inspecting the generated command

```bash
vllm-serve --print-cmd   # print the vllm serve argv to stderr, then run it
vllm-serve --dry-run     # print the argv and exit without running
```

### Verbose logging

```bash
VLLM_LOGGING_LEVEL=DEBUG flox activate --start-services
```

## File structure

```
vllm-runtime/
  .flox/env/manifest.toml   # Flox manifest (packages, on-activate hook, service)
  .flox/cache/vllm-config.yaml  # vLLM server config (auto-copied from package on first run)
  k8s/                       # Kubernetes manifests (Flox uncontained pattern)
  models/                    # Model cache (created on activation)
  examples/                  # Demo scripts (flox-bundled and HF-cached models)
  README.md                  # This file
```

Scripts (`vllm-preflight`, `vllm-resolve-model`, `vllm-serve`) are provided by the `flox/vllm-flox-runtime` package and available on `PATH` after activation. They are not stored in this directory. The default `config.yaml` (gpu-memory-utilization, dtype, logging) is bundled in the package and auto-copied to `.flox/cache/vllm-config.yaml` on first run — edit that copy to customize.

## Security notes

The runtime scripts handle untrusted input (model names, env files, lock files) and apply defense-in-depth.

### Env file trust model

The model env file is a trust boundary. In safe mode (default), `vllm-serve` parses the file with a restrictive Python parser that rejects shell interpolation and command substitution. In trusted mode, the file is `source`d directly — only enable this for env files you control.

Even in safe mode, the env file can set arbitrary environment variables (e.g., `PATH`, `LD_LIBRARY_PATH`, `HF_HOME`), so protect its location.

### File permissions

- **Env files**: written with `umask 077` and `chmod 600` — readable only by the owning user.
- **Lock files**: created with `umask 077`. Symlink safety is checked before opening.
- **Staging directories**: created under `$VLLM_MODELS_DIR/.staging/` with `umask 077`.

### Lockfile safety

- `vllm-preflight` validates the lockfile is not a symlink and is a regular file before opening.
- `vllm-resolve-model` uses per-model lock files (one per env file path) to prevent concurrent provisioning of the same model.
- The `mkdir`-based fallback includes stale PID detection to recover from crashes.
