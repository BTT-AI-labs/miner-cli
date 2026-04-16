# Modeldock

`modeldock` is a small Docker-based deployment helper for running large language models on a single Ubuntu host with NVIDIA GPUs.

It is intentionally narrow:

- no scheduler
- no cluster control plane
- no multi-tenant platform
- just config -> compose -> container -> health check

## What It Does

- checks whether the machine is ready for GPU containers
- generates a starter YAML config
- renders a Docker Compose deployment
- starts `sglang` or `vllm` containers
- runs a GPU container smoke test before deployment by default
- waits for `/v1/models` to become ready
- provides basic lifecycle commands for logs, status, stop, restart, and remove

## Install

```bash
cd modeldock
pip install -e .
```

For local development:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Quick Start

Generate a starter config:

```bash
modeldock init qwen35 \
  --engine sglang \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tp 8 \
  --port 8000
```

Run host checks:

```bash
modeldock doctor
```

Run host + deployment checks against a config:

```bash
modeldock doctor -f qwen35.yaml
```

Start the deployment:

```bash
export HF_TOKEN=hf_xxx
modeldock up -f qwen35.yaml
```

Skip the deployment-time smoke test if you have already verified Docker GPU access:

```bash
modeldock up -f qwen35.yaml --skip-smoke-test
```

Inspect the deployment:

```bash
modeldock status qwen35
modeldock logs qwen35 -f
```

Stop and remove:

```bash
modeldock stop qwen35
modeldock rm qwen35 --purge-files
```

## Example Config

See [examples/qwen72b_sglang.yaml](examples/qwen72b_sglang.yaml).

Important fields:

- `engine`: `sglang` or `vllm`
- `model`: Hugging Face model id
- `image`: Docker image to run
- `tensor_parallel`: number of GPUs to shard across
- `gpu_ids`: Docker GPU selector, usually `all`
- `hf_cache`: host path for model cache persistence
- `extra_args`: extra engine-specific CLI arguments
- `dcgm_exporter`: optional config block to add a DCGM metrics exporter service
- `miner_client`: optional config block to add the `miner-client` sidecar
- `metrics_collector`: optional config block to add a dedicated collector service built from your own image
- `extra_services`: arbitrary additional Docker Compose services appended under `services:`

Example:

```yaml
dcgm_exporter:
  enabled: true

metrics_collector:
  enabled: true
  image: your-registry/metrics-collector:latest
  listen_port: 8080
  host_port: 18080
  upstream_http_url: http://your-http-service.internal:9000/api
  environment:
    SCRAPE_INTERVAL: 15s
```

For the `miner-client` sidecar, use `miner_client`:

```yaml
miner_client:
  enabled: true
  image: your-registry/miner-client:latest
  listen_port: 7070
  host_port: 17070
  upstream_http_url: http://your-http-service.internal:9000/api
  environment:
    LOG_LEVEL: info
    MAIN_API_BASE_URL: https://main-api.example.com
    MINER_TOKEN: replace-me
    MINER_TARGET_MODEL: Qwen/Qwen2.5-72B-Instruct
```

When `miner_client.enabled` is `true`, modeldock generates a service in the same Docker Compose network and injects these environment variables by default:

- `MODELDOCK_DEPLOYMENT_NAME=<deployment-name>`
- `MODELDOCK_ENGINE=<engine>`
- `MODELDOCK_INFERENCE_BASE_URL=http://<deployment-name>:<port>`
- `MODELDOCK_OPENAI_BASE_URL=http://<deployment-name>:<port>/v1`
- `MODELDOCK_INFERENCE_METRICS_URL=http://<deployment-name>:<port>/metrics`
- `MINER_HTTP_HOST=0.0.0.0`
- `MINER_HTTP_PORT=<listen_port>`
- `MINER_VLLM_BASE_URL=http://<deployment-name>:<port>`

If `dcgm_exporter.enabled` is also `true`, modeldock also injects:

- `MODELDOCK_DCGM_EXPORTER_URL=http://dcgm-exporter:9400/metrics`
- `MINER_DCGM_METRICS_URL=http://dcgm-exporter:9400/metrics`

If `miner_client.upstream_http_url` is set, modeldock also injects:

- `UPSTREAM_HTTP_URL=<configured-url>`

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
- `miner_client.inference_metrics_path`
- `miner_client.openai_base_path`
- `miner_client.dcgm_metrics_path`
- `miner_client.upstream_http_url`
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

When `metrics_collector.enabled` is `true`, modeldock generates a collector service in the same Docker Compose network as the inference container and `dcgm-exporter`. The collector gets these environment variables by default:

- `INFERENCE_METRICS_URL=http://<deployment-name>:<port>/metrics`
- `DCGM_EXPORTER_URL=http://dcgm-exporter:9400/metrics`
- `MODELDOCK_DEPLOYMENT_NAME=<deployment-name>`
- `MODELDOCK_ENGINE=<engine>`
- `COLLECTOR_HTTP_HOST=0.0.0.0`
- `COLLECTOR_HTTP_PORT=<listen_port>`

If `metrics_collector.upstream_http_url` is set, modeldock also injects:

- `UPSTREAM_HTTP_URL=<configured-url>`

The collector exposes its own HTTP port inside the Compose network automatically. If you set `metrics_collector.host_port`, modeldock also publishes it on the host as `<host_port>:<listen_port>`.

You can override the generated defaults with:

- `metrics_collector.service_name`
- `metrics_collector.container_name`
- `metrics_collector.inference_metrics_path`
- `metrics_collector.dcgm_metrics_path`
- `metrics_collector.listen_host`
- `metrics_collector.listen_port`
- `metrics_collector.host_port`
- `metrics_collector.upstream_http_url`
- `metrics_collector.environment`
- `metrics_collector.command`
- `metrics_collector.entrypoint`
- `metrics_collector.volumes`
- `metrics_collector.ports`
- `metrics_collector.labels`
- `metrics_collector.depends_on`
- `metrics_collector.healthcheck`

Use `extra_services` only for unrelated sidecars that should not be coupled to the inference metrics flow.

## Notes

- This MVP assumes Docker, Docker Compose, NVIDIA drivers, and NVIDIA Container Toolkit are already installed.
- The default image names are placeholders that should be validated against the images you want to support in production.
- Deployment files are rendered into `~/.modeldock/deployments/<name>/`.
- `doctor` stays lightweight: Linux/Ubuntu basics, architecture, Docker daemon access, GPU inventory, `/dev/shm`, disk headroom, DNS, and config-specific fit such as open ports and tensor-parallel vs GPU count.
- `up` performs the heavier GPU container smoke test because that check pulls and runs a CUDA image.

## Engineering Notes

- Core config validation lives in `modeldock.config`, so YAML parsing and semantic checks are testable without invoking the CLI.
- Deployment rendering stays in `modeldock.deploy`, including compose-sidecar generation for observability services.
- CLI commands are intentionally thin wrappers around config, doctor, and deployment modules.
