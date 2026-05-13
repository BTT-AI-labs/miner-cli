# Miner CLI

`miner-cli` is a small Docker-based deployment helper for running large language models on a single Linux host with NVIDIA GPUs.

It is intentionally narrow:

- no scheduler
- no cluster control plane
- no multi-tenant platform
- just config -> compose -> container -> health check

## What It Does

- checks whether the machine is ready for GPU containers
- generates a starter YAML config
- renders a Docker Compose deployment
- starts `vllm` containers
- runs a GPU container smoke test before deployment by default
- waits for `/v1/models` to become ready
- provides basic lifecycle commands for logs, status, stop, restart, and remove

## Prerequisites

- Linux x86_64 host
- NVIDIA GPU visible on the host
- NVIDIA driver installed on the host
- Docker available on the host
- Python 3.10+
- `uv` if you want to use the recommended project workflow in this README

`miner-cli` does not install the full NVIDIA driver in V1. Driver installation and host GPU visibility remain host-level prerequisites.

## Install `uv`

This README uses `uv sync` and `uv run` as the primary workflow, so install `uv` first if it is not already available.

Recommended standalone installer on macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternative installation methods include:

```bash
pip install uv
```

After installation, reopen your shell if needed and confirm:

```bash
uv --version
```

Official `uv` installation docs: https://docs.astral.sh/uv/getting-started/installation/

## Install Miner CLI

### Option 1: Project workflow with `uv` (recommended)

```bash
cd miner-cli
uv sync
```

Run commands from the repo with:

```bash
uv run miner-cli doctor
```

### Option 2: Install the CLI into your Python environment with `pip`

The project exposes a console entrypoint in `pyproject.toml`, so you can install it directly:

```bash
cd miner-cli
pip install .
```

Then run:

```bash
miner-cli doctor
```

### Local development

```bash
uv sync --extra dev
uv run pytest
uv run --extra dev ruff check .
```

## Quick Start

Generate a starter config:

```bash
uv run miner-cli init qwen35 \
  --engine sglang \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tp 8 \
  --port 8000
```

For `vllm`, `init` currently uses the official upstream default image. If you explicitly want to keep the policy visible in config generation, you can set:

```bash
uv run miner-cli init qwen35 \
  --engine vllm \
  --model Qwen/Qwen2.5-72B-Instruct \
  --image-policy latest
```

When the generated or configured `vllm` image is `latest`, `miner-cli` now warns during `init`, `runtime prepare`, and `up` because floating upstream tags can change CUDA and driver requirements without notice.

Run host checks:

```bash
uv run miner-cli doctor
```

Prepare the host toolkit first if Docker or NVIDIA container support is not ready:

```bash
uv run miner-cli toolkit install
uv run miner-cli toolkit verify --smoke-test
```

Prepare the `vllm` runtime before deployment:

```bash
export HF_TOKEN=hf_xxx
uv run miner-cli runtime prepare --engine vllm
```

Run host + deployment checks against a config:

```bash
uv run miner-cli doctor -f qwen35.yaml
```

Start the deployment:

```bash
export HF_TOKEN=hf_xxx
uv run miner-cli up -f qwen35.yaml
```

Skip the deployment-time smoke test if you have already verified Docker GPU access:

```bash
uv run miner-cli up -f qwen35.yaml --skip-smoke-test
```

Inspect the deployment:

```bash
uv run miner-cli status qwen35
uv run miner-cli logs qwen35 -f
```

Stop and remove:

```bash
uv run miner-cli stop qwen35
uv run miner-cli rm qwen35 --purge-files
```

## Recommended Workflow

For a new miner host, use this order:

1. Install or verify local dependencies:

```bash
uv sync
```

2. Generate a starter config:

```bash
uv run miner-cli init qwen35 \
  --engine vllm \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tp 8 \
  --port 8000
```

3. Check the host:

```bash
uv run miner-cli doctor
```

4. If Docker or GPU container wiring is missing, prepare the host toolkit:

```bash
uv run miner-cli toolkit install
uv run miner-cli toolkit verify --smoke-test
```

5. Prepare the runtime for the selected engine and config:

```bash
export HF_TOKEN=hf_xxx
uv run miner-cli runtime prepare --engine vllm -f qwen35.yaml --smoke-test
```

6. Start the deployment:

```bash
uv run miner-cli up -f qwen35.yaml
```

7. Inspect status and logs if needed:

```bash
uv run miner-cli status qwen35
uv run miner-cli logs qwen35 -f
```

In practice, the commands have different responsibilities:

- `doctor`: lightweight host checks, and config checks if you specified config file by `-f`
- `toolkit install`: installs Docker-side prerequisites that the tool is allowed to manage
- `toolkit verify --smoke-test`: validates host GPU container readiness
- `runtime prepare`: validates image/runtime readiness for one engine and config
- `up`: deploys the actual workload and performs the final startup checks

## Troubleshooting

When `doctor`, `toolkit verify`, `runtime prepare`, or `up` fails, `miner-cli` prints a `Next steps` block after the main result table or error line. That block is the primary remediation guidance and is intended to tell the operator what to do next instead of only exposing raw command failures.

Common NVIDIA and deployment failure patterns:

- `nvidia-smi: not found`
  - Meaning: the host NVIDIA driver is not installed or `nvidia-smi` is not on `PATH`
  - Action: install the host NVIDIA driver first, confirm `nvidia-smi` works on the host, then rerun `uv run miner-cli toolkit verify`

- `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`
  - Meaning: the driver package may be present, but the kernel module or driver state is broken
  - Action: repair the host driver/module state, confirm `nvidia-smi` works, then rerun `uv run miner-cli toolkit verify`

- `gpu inventory: no GPUs detected`
  - Meaning: the driver is running, but no GPU is visible to the host
  - Action: check PCI visibility, VM passthrough, or cloud GPU attachment, then rerun `uv run miner-cli toolkit verify`

- `docker nvidia runtime: not configured`
  - Meaning: Docker is installed, but GPU runtime wiring is incomplete
  - Action: run `uv run miner-cli toolkit install`, then `uv run miner-cli toolkit verify --smoke-test`

- `gpu container smoke test` fails with messages like `driver version is insufficient` or `cuda>=...`
  - Meaning: the selected CUDA image requires a newer host NVIDIA driver
  - Action: upgrade the host driver or pin an older runtime image, then rerun `uv run miner-cli toolkit verify --smoke-test`

- `engine container smoke test` or `runtime smoke test` fails
  - Meaning: the image can be pulled, but the engine container still cannot start correctly with GPU access
  - Action: run `uv run miner-cli runtime prepare --engine vllm -f qwen35.yaml --smoke-test`

- `Image pull failed`
  - Meaning: the configured image tag may not exist, registry access may be broken, or authentication may be missing
  - Action: verify the configured image tag, confirm network access, and rerun `uv run miner-cli runtime prepare --engine vllm -f qwen35.yaml`

- `Container startup failed`
  - Meaning: Compose created the deployment, but the workload container could not boot successfully
  - Action: inspect logs with `uv run miner-cli logs <deployment-name> -f`, then rerun `uv run miner-cli runtime prepare --engine vllm -f <config> --smoke-test`

- readiness timeout
  - Meaning: the container is running, but `/v1/models` never became healthy in time
  - Action: inspect logs, verify model download progress and GPU memory fit, then rerun `uv run miner-cli runtime prepare --engine vllm -f <config> --smoke-test`

Operational guidance:

- `miner-cli` does not install the full NVIDIA driver in V1. The host driver remains a manual prerequisite.
- `toolkit install` is for Docker, permissions, NVIDIA Container Toolkit, and Docker runtime wiring.
- If host `nvidia-smi` works but the container smoke test fails, the problem is usually NVIDIA Container Toolkit or Docker runtime wiring, not the basic GPU hardware.
- #### If the container smoke test passes but the engine smoke test fails, the problem is usually image compatibility, CUDA/driver mismatch, or engine startup behavior.
- If you want more reproducible behavior, pin `image:` in your config instead of relying on floating `latest` tags.

## Example Config

See [examples/qwen72b_vllm.yaml](examples/qwen72b_vllm.yaml).

Important fields:

- `engine`: `sglang` or `vllm`. Only vllm supported right now.
- `model`: Hugging Face model id
- `image`: Docker image to run
- `tensor_parallel`: number of GPUs to shard across
- `gpu_ids`: Docker GPU selector, usually `all`
- `hf_cache`: host path for model cache persistence
- `extra_args`: extra engine-specific CLI arguments
- `dcgm_exporter`: optional config block to add a DCGM metrics exporter service
- `miner_client`: optional config block to add the `miner-client` sidecar
- `extra_services`: arbitrary additional Docker Compose services appended under `services:`

Example:

For the `dcgm-exporter` sidecar, use `dcgm-exporter`. It exposes an HTTP service on port `9400` with a `/metrics` endpoint for `Prometheus` collection.
```yaml
dcgm_exporter:
  enabled: true
  gpus: all
```

For the `miner-client` sidecar, use `miner_client`:

```yaml
miner_client:
  enabled: true
  image: your-registry/miner:latest
  volumes:
    - /data/minerhome:/root/.miner
  environment:
    LOG_LEVEL: info
    MAIN_API_BASE_URL: https://main-api.example.com
    MINER_TOKEN: replace-me
    MINER_TARGET_MODEL: Qwen/Qwen2.5-72B-Instruct
```

When `miner_client.enabled` is `true`, `miner-cli` generates a service in the same Docker Compose network and injects these environment variables by default:

- `MODELDOCK_DEPLOYMENT_NAME=<deployment-name>`

- `MINER_RUNTIME_TYPE=<engine>`
- `MINER_HTTP_HOST=0.0.0.0`
- `MINER_HTTP_PORT=<listen_port>`
- `MINER_VLLM_BASE_URL=http://<deployment-name>:<port>`

If `dcgm_exporter.enabled` is also `true`, `miner-cli` also injects:

- `MINER_DCGM_METRICS_URL=http://dcgm-exporter:9400/metrics`

For the `miner-client` project in this repo, you should usually provide these additional variables under `miner_client.environment`:

- `MAIN_API_BASE_URL`
- `MINER_TOKEN`
- `MINER_TARGET_MODEL`

You can override the generated defaults with:

- `miner_client.service_name`
- `miner_client.container_name`
- `miner_client.listen_host`
- `miner_client.listen_port`
- `miner_client.host_port`
- `miner_client.dcgm_metrics_path`
- `miner_client.environment`
- `miner_client.command`
- `miner_client.entrypoint`
- `miner_client.volumes`
- `miner_client.ports`
- `miner_client.labels`
- `miner_client.depends_on`
- `miner_client.healthcheck`

Backward compatibility:

- `custom_service` is still accepted as a legacy alias for `miner_client`
- do not set both fields in the same config

Use `extra_services` only for unrelated sidecars that should not be coupled to the inference metrics flow.

## Notes

- This MVP assumes Docker, Docker Compose, NVIDIA drivers, and NVIDIA Container Toolkit are already installed.
- `toolkit verify` is the most portable path and is intended to work across Linux distributions.
- `toolkit install` uses distro-family backends for `debian`, `rhel`, and `arch` style systems and performs privileged host changes through installer scripts.
- On unsupported distributions, `toolkit install` stops early and prints manual installation guidance instead of guessing.
- `toolkit verify` and `runtime prepare` are the recommended preparation steps before `up` when a host is not already ready.
- Failing `doctor`, `toolkit verify`, and `runtime prepare` runs now print a `Next steps` block that maps each failed check to the most relevant remediation command or manual host action.
- `miner-cli` should only default to tags that are confirmed to exist upstream. Until a verified stable tag policy is maintained in-repo, the generated `vllm` image remains the official upstream default.
- If you need stricter reproducibility, set `image:` explicitly in your config instead of relying on the generated default.
- Deployment files are rendered into `~/.miner-cli/deployments/<name>/`.
- `doctor` stays lightweight: Linux/Ubuntu basics, architecture, Docker daemon access, GPU inventory, `/dev/shm`, disk headroom, DNS, and config-specific fit such as open ports and tensor-parallel vs GPU count.
- `up` performs the heavier GPU container smoke test because that check pulls and runs a CUDA image.

## Engineering Notes

- Core config validation lives in `miner_cli.config`, so YAML parsing and semantic checks are testable without invoking the CLI.
- Deployment rendering stays in `miner_cli.deploy`, including compose-sidecar generation for observability services.
- CLI commands are intentionally thin wrappers around config, doctor, and deployment modules.
