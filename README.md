# vLLM Runtime Environment

Production vLLM inference server packaged as a composable Flox environment. Part of a three-environment reference architecture for serving LLMs with monitoring and secure network access.

Developed on an RTX 5090 (24GB, SM120), but the architecture applies to any NVIDIA GPU setup — single or multi-GPU. Adjust `config.yaml`, the SM variant in the manifest, and the parallelism settings for your hardware.

- **vLLM**: 0.15.1
- **CUDA**: 12.9 (requires NVIDIA driver 560+)
- **Target**: SM120 (RTX 5090 / Blackwell)
- **Python**: 3.12

## Quick Start

```bash
# Activate the environment
flox activate -d /home/daedalus/dev/vllm-runtime

# Start the vLLM service (downloads the default model on first run)
flox services start

# Or activate and start services in one step
flox activate -d /home/daedalus/dev/vllm-runtime --start-services
```

The server binds to `127.0.0.1:8000` by default. The default model is `meta-llama/Llama-3.1-8B-Instruct`.

## Swapping Models

vLLM serves one model at a time per process. To switch models:

**Option 1: Override at activation time**

```bash
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3 flox activate -d /home/daedalus/dev/vllm-runtime --start-services
```

**Option 2: Edit the manifest and restart**

```bash
# Edit VLLM_MODEL in .flox/env/manifest.toml [vars] section
flox services restart vllm
```

### Model Sizing Reference

What fits depends on total VRAM across all GPUs used for tensor parallelism.

**Single GPU (24GB — e.g., RTX 5090, RTX 4090, A10G):**

| Model | Precision | Approx max-model-len |
|-------|-----------|---------------------|
| Llama 3.1 8B / Mistral 7B / Qwen 2.5 7B | BF16 (full) | up to 32K |
| Llama 3.1 13B / Qwen 2.5 14B | AWQ/GPTQ 4-bit | up to 16K |
| Llama 3.1 70B | AWQ 4-bit | ~4K |

**2x GPUs (48GB total — TP=2, e.g., 2× RTX 5090):**

| Model | Precision | Approx max-model-len |
|-------|-----------|---------------------|
| Llama 3.1 8B / Mistral 7B | BF16 (full) | up to native max |
| Llama 3.1 13B / Qwen 2.5 14B | BF16 (full) | up to 32K |
| Llama 3.1 70B | AWQ 4-bit | up to 16K |

**4x GPUs (96GB total — TP=4, e.g., 4× RTX 5090):**

| Model | Precision | Approx max-model-len |
|-------|-----------|---------------------|
| Llama 3.1 13B / Qwen 2.5 14B | BF16 (full) | up to native max |
| Llama 3.1 70B | AWQ 4-bit | up to 32K+ |

**Larger configurations (e.g., 2× A100 80GB = 160GB, 8× H100 80GB = 640GB):**

| GPUs | Model | Precision | Approx max-model-len |
|------|-------|-----------|---------------------|
| 2× A100 80GB (TP=2) | Llama 3.1 70B | BF16 (full) | up to 8K |
| 4× A100 80GB (TP=4) | Llama 3.1 70B | BF16 (full) | up to 32K |
| 8× H100 80GB (TP=8) | Llama 3.1 405B | AWQ 4-bit | up to 16K |
| 8× H100 80GB (TP=8) | Llama 3.1 405B | FP8 | up to 8K |

Note: Llama 3.1 70B at BF16 requires ~140GB for weights alone — it needs at least 2× 80GB GPUs or 8× 24GB GPUs. Llama 3.1 405B at AWQ 4-bit requires ~202GB — it needs at least 4× 80GB GPUs.

To increase `max-model-len` for smaller models, edit `config.yaml`.

## Multi-GPU (Tensor and Pipeline Parallelism)

This environment supports single-machine multi-GPU via two parallelism modes, controlled by environment variables in the manifest:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Shard weight matrices within each layer across N GPUs (most common) |
| `VLLM_PIPELINE_PARALLEL_SIZE` | `1` | Distribute layers across N GPUs in pipeline stages (for very large models) |

**Tensor parallelism (TP)** is the standard approach — it shards each layer's weights across GPUs and requires NVLink or PCIe for inter-GPU communication. Use TP when all GPUs are identical and directly connected.

**Pipeline parallelism (PP)** assigns different layers to different GPUs sequentially. It uses less inter-GPU bandwidth but adds pipeline bubbles. Use PP when GPUs differ in size or when combining with TP (e.g., TP=2 PP=2 for 4 GPUs).

### Examples

```bash
# 2x GPU tensor parallel
VLLM_TENSOR_PARALLEL_SIZE=2 flox activate -d /home/daedalus/dev/vllm-runtime --start-services

# 4x GPU tensor parallel
VLLM_TENSOR_PARALLEL_SIZE=4 flox activate -d /home/daedalus/dev/vllm-runtime --start-services

# 4x GPU hybrid: 2-way tensor parallel × 2-way pipeline parallel
VLLM_TENSOR_PARALLEL_SIZE=2 VLLM_PIPELINE_PARALLEL_SIZE=2 flox activate -d /home/daedalus/dev/vllm-runtime --start-services
```

Or set the defaults in the manifest `[vars]` section and restart:

```bash
flox services restart vllm
```

### Constraints

- TP size must evenly divide the model's attention heads (e.g., Llama 70B has 64 heads — TP=1, 2, 4, 8 all work; TP=3 doesn't)
- TP × PP must equal the total number of GPUs you want to use
- All GPUs in a TP group should be the same model (mixed GPU types cause the slowest GPU to bottleneck)
- `gpu-memory-utilization` in `config.yaml` applies per GPU

### Multi-Node (Distributed)

For deployments spanning multiple machines, vLLM uses Ray for distributed execution. This is a fundamentally different topology from single-node multi-GPU and is outside the scope of `vllm-runtime/`.

**How vLLM uses Ray for multi-node:**

On a single machine, vLLM uses Python multiprocessing + NCCL for GPU communication — no Ray involved. When a model is too large for one machine's GPUs (or you want to scale throughput across nodes), vLLM delegates to Ray:

- **Ray head node** — runs the Ray GCS (Global Control Store), the dashboard, and coordinates worker placement. The vLLM API server runs here.
- **Ray worker nodes** — each node runs a Ray worker process that spawns vLLM engine workers. Each worker owns one or more local GPUs.
- **NCCL across nodes** — the actual tensor communication uses NCCL over InfiniBand or RoCE (RDMA over Converged Ethernet), not Ray's object store. Ray just gets the processes running and connected; NCCL handles the fast path.

A standalone `vllm-distributed/` Flox environment would need:
- `ray` package alongside `vllm`
- Separate services for Ray head vs. Ray worker roles (or role selection via an env var)
- Network configuration: `RAY_ADDRESS`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` for InfiniBand
- Shared model storage (NFS mount, S3 with `HF_HUB_OFFLINE=1`, or pre-downloaded per node)
- Per-node GPU visibility: `CUDA_VISIBLE_DEVICES` or resource scheduling

#### Multi-Node on Kubernetes with KubeRay + Flox Uncontained

The production path for multi-node vLLM is Kubernetes with the [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html) — and Flox's [Kubernetes, Uncontained](https://flox.dev/blog/kubernetes-uncontained) pattern eliminates the biggest pain point: building and distributing multi-gigabyte CUDA container images.

**Why this combination works:**

KubeRay manages the Ray cluster lifecycle — it creates `RayCluster` CRDs that define head and worker pod templates, handles autoscaling based on pending Ray tasks, and manages failover. Normally, each pod pulls a 15-30GB container image containing CUDA, PyTorch, vLLM, and the model weights.

With Flox Uncontained, those pods use `runtimeClassName: flox` instead. The Flox containerd shim realizes the environment from a Flox environment reference (a single annotation), pulling only the packages that aren't already in the node-local immutable store. On a warm node — one that has previously run a vLLM pod — startup is near-instant because CUDA libraries, PyTorch, and vLLM are already cached.

This is significant for GPU workloads specifically:

- **No image bloat** — CUDA toolkit (~3GB), PyTorch (~2GB), and vLLM with compiled kernels (~1GB) are stored once per node in hash-addressed paths, not duplicated across image layers
- **Fast scale-up** — when KubeRay autoscaler adds worker pods, they mount cached dependencies immediately instead of pulling a fresh 20GB image
- **Atomic dependency updates** — updating vLLM or a CUDA dependency is a one-line Flox manifest change pushed to FloxHub, not an image rebuild + registry push + rolling restart
- **Same environment dev-to-prod** — the same Flox environment that runs on a developer's workstation (this `vllm-runtime/`) runs inside Kubernetes pods, bit-for-bit identical

**Architecture on Kubernetes:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                             │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  KubeRay Operator                                         │  │
│  │  Manages RayCluster CRD, autoscaling, failover            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │  Ray Head Pod        │  │  Ray Worker Pod (×N) │              │
│  │  runtimeClass: flox  │  │  runtimeClass: flox  │              │
│  │                      │  │                      │              │
│  │  ┌────────────────┐  │  │  ┌────────────────┐  │              │
│  │  │ Flox env:       │  │  │  │ Flox env:       │  │              │
│  │  │  vllm + ray     │  │  │  │  vllm + ray     │  │              │
│  │  │  cuda 12.9      │  │  │  │  cuda 12.9      │  │              │
│  │  │  python 3.12    │  │  │  │  python 3.12    │  │              │
│  │  └────────────────┘  │  │  └────────────────┘  │              │
│  │                      │  │                      │              │
│  │  vLLM API server     │  │  vLLM engine worker  │              │
│  │  Ray GCS + dashboard │  │  NCCL ←──────────────┤── IB/RoCE   │
│  │  Port 8000           │  │  GPU: 0..M           │              │
│  └─────────────────────┘  └─────────────────────┘              │
│                                                                 │
│  Node-local Flox store: /nix/store/<hash>-vllm-0.15.1/...      │
│  (shared read-only across all pods on the node)                 │
└─────────────────────────────────────────────────────────────────┘
```

**Example KubeRay RayCluster with Flox:**

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: vllm-cluster
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      metadata:
        annotations:
          flox.dev/environment: "myorg/vllm-runtime:latest"
      spec:
        runtimeClassName: flox
        containers:
          - name: ray-head
            image: flox/empty:1.0.0
            command: ["vllm", "serve", "meta-llama/Llama-3.1-70B-Instruct",
                      "--tensor-parallel-size", "8",
                      "--config", "/app/config.yaml"]
            resources:
              limits:
                nvidia.com/gpu: "0"   # head doesn't need a GPU
            ports:
              - containerPort: 8000   # vLLM API
              - containerPort: 6379   # Ray GCS
              - containerPort: 8265   # Ray dashboard
        nodeSelector:
          flox.dev/enabled: "true"

  workerGroupSpecs:
    - replicas: 2
      minReplicas: 1
      maxReplicas: 4
      groupName: gpu-workers
      rayStartParams: {}
      template:
        metadata:
          annotations:
            flox.dev/environment: "myorg/vllm-runtime:latest"
        spec:
          runtimeClassName: flox
          containers:
            - name: ray-worker
              image: flox/empty:1.0.0
              resources:
                limits:
                  nvidia.com/gpu: "4"
          nodeSelector:
            flox.dev/enabled: "true"
            nvidia.com/gpu.product: "NVIDIA-H100-80GB-HBM3"
```

Key details in this manifest:
- `image: flox/empty:1.0.0` — the 49-byte stub image required by CRI/OCI; the actual environment comes from the `flox.dev/environment` annotation
- `flox.dev/environment` — references the Flox environment published to FloxHub; this is the same environment definition as `vllm-runtime/`, extended with `ray`
- `runtimeClassName: flox` — routes pod startup through the Flox containerd shim
- Worker pods request 4 GPUs each via `nvidia.com/gpu`; the NVIDIA device plugin and KubeRay handle GPU assignment
- KubeRay autoscaler can add workers (up to `maxReplicas`) when vLLM's request queue grows

**Flox environment for Kubernetes multi-node (`vllm-distributed/manifest.toml`):**

```toml
version = 1

[install]
vllm-python312-cuda12_9-sm90.pkg-path = "flox/vllm-python312-cuda12_9-sm90"
vllm-python312-cuda12_9-sm90.pkg-group = "vllm-python312-cuda12_9-sm90"
ray.pkg-path = "ray"

[vars]
VLLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
VLLM_HOST = "0.0.0.0"
VLLM_PORT = "8000"
VLLM_TENSOR_PARALLEL_SIZE = "8"
VLLM_PIPELINE_PARALLEL_SIZE = "1"
NCCL_SOCKET_IFNAME = "eth0"
HF_HOME = "/models"
```

This environment would be published to FloxHub and referenced by the KubeRay pod templates. The SM variant (`sm90` for H100 in this example) would be chosen based on the target GPU hardware in the cluster.

## Downloading Models for Offline Use

Pre-download a model so it's available without internet access:

```bash
flox activate -d /home/daedalus/dev/vllm-runtime
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

Models are cached in `./models/` (the `HF_HOME` directory).

## API Endpoints

The server exposes an OpenAI-compatible API:

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming supported) |
| `POST /v1/completions` | Text completions |
| `GET /v1/models` | List loaded models |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |

### Example: Chat Completion

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm-local-dev" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

### Example: Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Configuration

Server settings are in `config.yaml`. Key parameters:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `gpu-memory-utilization` | 0.92 | Per-GPU VRAM fraction for KV cache. 0.92 for 24GB, 0.95 for 48GB+ |
| `max-model-len` | 8192 | Conservative default; increase for smaller models or more VRAM |
| `dtype` | auto | BF16 for BF16 models, FP16 for FP16/FP32 |
| `host` | 127.0.0.1 | Localhost only; use the proxy env for external access |

Parallelism and model selection are environment variables in `.flox/env/manifest.toml` `[vars]`:

| Variable | Default | Notes |
|----------|---------|-------|
| `VLLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID or local path |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `VLLM_PIPELINE_PARALLEL_SIZE` | `1` | Number of GPUs for pipeline parallelism |
| `VLLM_API_KEY` | `sk-vllm-local-dev` | Bearer token for API authentication |
| `HF_HOME` | `$FLOX_ENV_PROJECT/models` | HuggingFace model cache directory |

## Service Management

```bash
# Check status
flox services status

# View logs
flox services logs vllm

# Restart after config changes
flox services restart vllm

# Stop
flox services stop
```

## Composable Architecture

This environment is one layer of a three-environment reference stack. Each environment is self-contained and independently useful, but they compose via Flox `[include]` for production deployments.

```
┌─────────────────────────────────────────────────┐
│  vllm-proxy/          (Layer 3: Network Edge)   │
│  nginx reverse proxy, TLS, rate limiting        │
│  Listens: 0.0.0.0:443 (HTTPS), :80 (redirect)  │
│  Proxies to: 127.0.0.1:8000 (vllm)             │
│  includes: vllm-monitoring/                     │
├─────────────────────────────────────────────────┤
│  vllm-monitoring/     (Layer 2: Observability)  │
│  Prometheus + Grafana                           │
│  Prometheus: 127.0.0.1:9090                     │
│  Grafana:    127.0.0.1:3000                     │
│  Scrapes:    127.0.0.1:8000/metrics             │
│  includes: vllm-runtime/                        │
├─────────────────────────────────────────────────┤
│  vllm-runtime/        (Layer 1: Inference)      │
│  vLLM server, CUDA, model management            │
│  Listens: 127.0.0.1:8000                        │
└─────────────────────────────────────────────────┘
```

Composition uses Flox `[include]` — each layer inherits the services and environment of the layer below:

```toml
# In vllm-monitoring/manifest.toml
[include]
environments = [{ dir = "../vllm-runtime" }]

# In vllm-proxy/manifest.toml
[include]
environments = [{ dir = "../vllm-monitoring" }]
```

Activating `vllm-proxy/` starts all three layers. Activating `vllm-monitoring/` starts inference + monitoring without the proxy. Activating `vllm-runtime/` alone runs just the inference server.

### Deployment Modes

| Mode | Activate | Use Case |
|------|----------|----------|
| **Local dev** | `vllm-runtime/` | Direct localhost access, no overhead |
| **Monitored** | `vllm-monitoring/` | Local dev with Prometheus/Grafana dashboards |
| **Production** | `vllm-proxy/` | Full stack: TLS, auth, rate limiting, monitoring |

---

### Environment: `vllm-monitoring/`

Prometheus scrapes vLLM's `/metrics` endpoint. Grafana visualizes the metrics with a pre-configured vLLM dashboard.

**Packages:**
- `prometheus` — metrics collection and storage
- `grafana` — dashboard UI

**Services:**

| Service | Bind Address | Purpose |
|---------|-------------|---------|
| `prometheus` | `127.0.0.1:9090` | Scrapes vLLM `/metrics` every 15s |
| `grafana` | `127.0.0.1:3000` | Dashboard UI, datasource auto-configured |

**Key files:**

| File | Purpose |
|------|---------|
| `prometheus.yml` | Scrape config targeting `127.0.0.1:8000/metrics` |
| `grafana/provisioning/datasources/prometheus.yaml` | Auto-registers Prometheus as a Grafana datasource |
| `grafana/provisioning/dashboards/dashboard.yaml` | Dashboard provisioning config |
| `grafana/dashboards/vllm.json` | Pre-built vLLM dashboard (request rate, latency p50/p95/p99, GPU memory, KV cache usage, batch size, token throughput) |

**Prometheus scrape config (`prometheus.yml`):**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "vllm"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

**vLLM metrics available for dashboards:**

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:num_requests_running` | Gauge | Currently processing requests |
| `vllm:num_requests_waiting` | Gauge | Requests queued |
| `vllm:gpu_cache_usage_perc` | Gauge | KV cache utilization (0.0–1.0) |
| `vllm:avg_generation_throughput_toks_per_s` | Gauge | Token generation throughput |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request latency |
| `vllm:time_to_first_token_seconds` | Histogram | Time to first token (TTFT) |
| `vllm:time_per_output_token_seconds` | Histogram | Inter-token latency (ITL) |
| `vllm:num_preemptions_total` | Counter | Preempted requests (memory pressure) |

**Manifest structure (`vllm-monitoring/.flox/env/manifest.toml`):**

Note: Flox `[vars]` are literal strings — they can't reference other variables like `$FLOX_ENV_PROJECT`. Grafana reads `GF_*` environment variables at runtime, so paths that need to be portable are set in the `[hook]` section using shell expansion instead.

```toml
version = 1

[install]
prometheus.pkg-path = "prometheus"
grafana.pkg-path = "grafana"

[vars]
PROMETHEUS_HOST = "127.0.0.1"
PROMETHEUS_PORT = "9090"
GRAFANA_HOST = "127.0.0.1"
GRAFANA_PORT = "3000"
GF_SECURITY_ADMIN_PASSWORD = "admin"
GF_SERVER_HTTP_ADDR = "127.0.0.1"
GF_SERVER_HTTP_PORT = "3000"

[hook]
on-activate = '''
  mkdir -p "$FLOX_ENV_PROJECT/grafana/data"
  mkdir -p "$FLOX_ENV_PROJECT/grafana/provisioning/datasources"
  mkdir -p "$FLOX_ENV_PROJECT/grafana/provisioning/dashboards"
  mkdir -p "$FLOX_ENV_PROJECT/grafana/dashboards"
  mkdir -p "$FLOX_ENV_PROJECT/prometheus/data"

  # Set Grafana paths via shell expansion (can't use $FLOX_ENV_PROJECT in [vars])
  export GF_PATHS_PROVISIONING="$FLOX_ENV_PROJECT/grafana/provisioning"
  export GF_PATHS_DATA="$FLOX_ENV_PROJECT/grafana/data"
'''

[services]
prometheus.command = "prometheus --config.file=$FLOX_ENV_PROJECT/prometheus.yml --storage.tsdb.path=$FLOX_ENV_PROJECT/prometheus/data --web.listen-address=$PROMETHEUS_HOST:$PROMETHEUS_PORT"
grafana.command = "grafana server --homepath=$(dirname $(dirname $(which grafana)))/share/grafana --config=$FLOX_ENV_PROJECT/grafana/grafana.ini"

[include]
environments = [{ dir = "../vllm-runtime" }]
```

---

### Environment: `vllm-proxy/`

nginx reverse proxy that sits in front of the vLLM server. Handles TLS termination, bearer token authentication, rate limiting, and request logging. This is the only layer that binds to external network interfaces.

**Why a reverse proxy in front of vLLM:**

- **TLS termination** — vLLM's uvicorn server doesn't handle TLS natively. Any client expecting `https://` (including the OpenAI Python SDK with a custom base URL) needs a TLS-terminating proxy.
- **Rate limiting** — protects the GPU from being overwhelmed. A single long-context request can monopolize the GPU for seconds; rate limiting prevents queue starvation.
- **IP allowlisting** — restrict access to known clients or CIDR ranges without modifying vLLM.
- **Request/response logging** — structured access logs separate from vLLM's application logs, useful for auditing and billing.
- **Health check abstraction** — nginx can return 503 during model loads or restarts, giving clients a clear signal rather than connection refused.
- **Future: load balancing** — if the deployment scales to multiple GPUs or machines, nginx distributes requests across vLLM instances without changing client configuration.

For local-only use (localhost clients, no network exposure), this layer is optional — `vllm-runtime/` alone is sufficient.

**Packages:**
- `nginx` — reverse proxy and TLS termination

**Services:**

| Service | Bind Address | Purpose |
|---------|-------------|---------|
| `nginx` | `0.0.0.0:443` (HTTPS), `0.0.0.0:80` (HTTP redirect) | TLS termination, auth, rate limiting |

**Key files:**

| File | Purpose |
|------|---------|
| `nginx.conf` | Main nginx configuration |
| `certs/server.crt` | TLS certificate (self-signed for dev, replace for production) |
| `certs/server.key` | TLS private key |

**nginx configuration (`nginx.conf`):**

```nginx
worker_processes auto;
error_log /home/daedalus/dev/vllm-proxy/logs/error.log warn;
pid /home/daedalus/dev/vllm-proxy/logs/nginx.pid;

events {
    worker_connections 256;
}

http {
    log_format json_combined escape=json '{'
        '"time":"$time_iso8601",'
        '"remote_addr":"$remote_addr",'
        '"request":"$request",'
        '"status":$status,'
        '"body_bytes_sent":$body_bytes_sent,'
        '"request_time":$request_time,'
        '"upstream_response_time":"$upstream_response_time"'
    '}';

    access_log /home/daedalus/dev/vllm-proxy/logs/access.log json_combined;

    # Rate limiting: 10 requests/sec per client IP, burst of 20
    limit_req_zone $binary_remote_addr zone=vllm_limit:10m rate=10r/s;

    # Upstream vLLM server
    upstream vllm_backend {
        server 127.0.0.1:8000;
        keepalive 32;
    }

    # HTTP -> HTTPS redirect
    server {
        listen 80;
        return 301 https://$host$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl;

        ssl_certificate     /home/daedalus/dev/vllm-proxy/certs/server.crt;
        ssl_certificate_key /home/daedalus/dev/vllm-proxy/certs/server.key;
        ssl_protocols       TLSv1.2 TLSv1.3;
        ssl_ciphers         HIGH:!aNULL:!MD5;

        # Health check endpoint (no auth, no rate limit)
        location /health {
            proxy_pass http://vllm_backend/health;
            proxy_set_header Host $host;
        }

        # Prometheus metrics (restrict to localhost/monitoring)
        location /metrics {
            allow 127.0.0.1;
            deny all;
            proxy_pass http://vllm_backend/metrics;
        }

        # OpenAI-compatible API
        location /v1/ {
            limit_req zone=vllm_limit burst=20 nodelay;

            proxy_pass http://vllm_backend/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Streaming support
            proxy_buffering off;
            proxy_cache off;
            proxy_http_version 1.1;
            proxy_set_header Connection "";

            # Long timeout for generation
            proxy_read_timeout 300s;
        }
    }
}
```

**Manifest structure (`vllm-proxy/.flox/env/manifest.toml`):**

```toml
version = 1

[install]
nginx.pkg-path = "nginx"
openssl.pkg-path = "openssl"

[vars]
NGINX_HOST = "0.0.0.0"
NGINX_HTTPS_PORT = "443"
NGINX_HTTP_PORT = "80"
VLLM_PROXY_RATE_LIMIT = "10r/s"
VLLM_PROXY_RATE_BURST = "20"

[hook]
on-activate = '''
  mkdir -p "$FLOX_ENV_PROJECT/logs"
  mkdir -p "$FLOX_ENV_PROJECT/certs"

  # Generate self-signed cert if none exists
  if [ ! -f "$FLOX_ENV_PROJECT/certs/server.crt" ]; then
    echo "Generating self-signed TLS certificate..."
    openssl req -x509 -newkey rsa:2048 -nodes \
      -keyout "$FLOX_ENV_PROJECT/certs/server.key" \
      -out "$FLOX_ENV_PROJECT/certs/server.crt" \
      -days 365 -subj "/CN=localhost" 2>/dev/null
    echo "  Self-signed cert created. Replace with a real cert for production."
  fi
'''

[services]
nginx.command = "nginx -c $FLOX_ENV_PROJECT/nginx.conf -g 'daemon off;'"

[include]
environments = [{ dir = "../vllm-monitoring" }]
```

---

## File Structure

The complete three-environment layout:

```
vllm-runtime/                       # Layer 1: Inference
  .flox/env/manifest.toml
  config.yaml                       # vLLM server configuration
  models/                           # HuggingFace model cache
  README.md

vllm-monitoring/                    # Layer 2: Observability
  .flox/env/manifest.toml
  prometheus.yml                    # Scrape config
  prometheus/data/                  # Prometheus TSDB storage
  grafana/
    grafana.ini                     # Grafana server config
    provisioning/
      datasources/prometheus.yaml   # Auto-registers Prometheus
      dashboards/dashboard.yaml     # Dashboard provisioning
    dashboards/
      vllm.json                     # Pre-built vLLM dashboard
  README.md

vllm-proxy/                         # Layer 3: Network Edge
  .flox/env/manifest.toml
  nginx.conf                        # Reverse proxy configuration
  certs/
    server.crt                      # TLS certificate
    server.key                      # TLS private key
  logs/                             # Access and error logs
  README.md
```
