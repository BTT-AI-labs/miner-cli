from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml
from jinja2 import Template

from .config import DEFAULT_DEPLOYMENTS_DIR, DeploymentConfig, default_image_for_engine
from .preparation import ProgressLogger


@dataclass
class DeploymentPaths:
    root: Path
    config_path: Path
    compose_path: Path
    env_path: Path


def deployment_paths(name: str, base_dir: Path | None = None) -> DeploymentPaths:
    root = (base_dir or DEFAULT_DEPLOYMENTS_DIR) / name
    return DeploymentPaths(
        root=root,
        config_path=root / "config.yaml",
        compose_path=root / "compose.yaml",
        env_path=root / ".env",
    )


def default_image(engine: str) -> str:
    return default_image_for_engine(engine)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_launch_args(config: DeploymentConfig) -> list[str]:
    if config.engine == "sglang":
        args = [
            "python",
            "-m",
            "sglang.launch_server",
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--model-path",
            config.model,
            "--tp",
            str(config.tensor_parallel),
        ]
        if config.dtype:
            args.extend(["--dtype", config.dtype])
        if config.trust_remote_code:
            args.append("--trust-remote-code")
        if config.max_model_len:
            args.extend(["--context-length", str(config.max_model_len)])
        if config.api_key:
            args.extend(["--api-key", config.api_key])
    else:
        args = [
            config.model,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--tensor-parallel-size",
            str(config.tensor_parallel),
        ]
        if config.trust_remote_code:
            args.append("--trust-remote-code")
        if config.dtype:
            args.extend(["--dtype", config.dtype])
        if config.max_model_len:
            args.extend(["--max-model-len", str(config.max_model_len)])
        if config.api_key:
            args.extend(["--api-key", config.api_key])

    args.extend(config.extra_args)
    return args


def build_launch_command(config: DeploymentConfig) -> str:
    args = build_launch_args(config)
    return " ".join(shell_quote(item) for item in args)


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def _render_service_block(service_name: str, service: dict[str, object]) -> str:
    service_yaml = yaml.safe_dump(
        {service_name: service},
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
    return _indent_block(service_yaml, 2)


def _build_dcgm_exporter_service(config: DeploymentConfig) -> dict[str, object] | None:
    settings = config.dcgm_exporter
    if not settings.get("enabled"):
        return None

    service: dict[str, object] = {
        "image": settings.get(
            "image",
            "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04",
        ),
        "container_name": settings.get("container_name", f"{config.name}-dcgm-exporter"),
        "restart": settings.get("restart", "unless-stopped"),
        "cap_add": settings.get("cap_add", ["SYS_ADMIN"]),
        "ports": settings.get("ports", ["9400:9400"]),
    }

    for key in ("gpus", "environment", "volumes", "command"):
        if key in settings:
            service[key] = settings[key]

    return service


def _build_miner_client_service(config: DeploymentConfig) -> tuple[str, dict[str, object]] | None:
    settings = config.miner_client
    if not settings.get("enabled"):
        return None

    image = settings.get("image")
    if not image:
        raise ValueError("miner_client.image is required when miner_client.enabled=true")

    service_name = str(settings.get("service_name", "miner-client"))
    listen_host = str(settings.get("listen_host", "0.0.0.0"))
    listen_port = int(settings.get("listen_port", 8080))
    inference_base_url = f"http://{config.name}:{config.port}"
    inference_metrics_path = str(settings.get("inference_metrics_path", "/metrics"))
    openai_base_path = str(settings.get("openai_base_path", "/v1"))
    dcgm_metrics_path = str(settings.get("dcgm_metrics_path", "/metrics"))
    environment = dict(settings.get("environment", {}))
    environment.setdefault("MODELDOCK_DEPLOYMENT_NAME", config.name)
    environment.setdefault("MODELDOCK_ENGINE", config.engine)
    environment.setdefault("MODELDOCK_INFERENCE_BASE_URL", inference_base_url)
    environment.setdefault("MODELDOCK_OPENAI_BASE_URL", f"{inference_base_url}{openai_base_path}")
    environment.setdefault(
        "MODELDOCK_INFERENCE_METRICS_URL",
        f"{inference_base_url}{inference_metrics_path}",
    )
    environment.setdefault("MINER_HTTP_HOST", listen_host)
    environment.setdefault("MINER_HTTP_PORT", str(listen_port))
    environment.setdefault("MINER_VLLM_BASE_URL", inference_base_url)
    if config.dcgm_exporter.get("enabled"):
        environment.setdefault(
            "MODELDOCK_DCGM_EXPORTER_URL",
            f"http://dcgm-exporter:9400{dcgm_metrics_path}",
        )
        environment.setdefault(
            "MINER_DCGM_METRICS_URL",
            f"http://dcgm-exporter:9400{dcgm_metrics_path}",
        )

    upstream_http_url = settings.get("upstream_http_url")
    if upstream_http_url:
        environment.setdefault("UPSTREAM_HTTP_URL", str(upstream_http_url))

    depends_on = [config.name]
    if config.dcgm_exporter.get("enabled"):
        depends_on.append("dcgm-exporter")
    extra_depends_on = settings.get("depends_on", [])
    if extra_depends_on:
        depends_on.extend(str(item) for item in extra_depends_on)

    service: dict[str, object] = {
        "image": image,
        "container_name": settings.get("container_name", f"{config.name}-{service_name}"),
        "restart": settings.get("restart", "unless-stopped"),
        "depends_on": depends_on,
        "environment": environment,
        "expose": [str(listen_port)],
    }

    publish_port = settings.get("host_port")
    if publish_port is not None and "ports" not in settings:
        service["ports"] = [f"{publish_port}:{listen_port}"]

    for key in ("ports", "volumes", "command", "entrypoint", "labels", "gpus", "healthcheck"):
        if key in settings:
            service[key] = settings[key]

    return service_name, service


def _build_metrics_collector_service(
    config: DeploymentConfig,
) -> tuple[str, dict[str, object]] | None:
    settings = config.metrics_collector
    if not settings.get("enabled"):
        return None

    image = settings.get("image")
    if not image:
        raise ValueError("metrics_collector.image is required when metrics_collector.enabled=true")

    service_name = str(settings.get("service_name", "metrics-collector"))
    inference_metrics_path = str(settings.get("inference_metrics_path", "/metrics"))
    dcgm_metrics_path = str(settings.get("dcgm_metrics_path", "/metrics"))
    listen_host = str(settings.get("listen_host", "0.0.0.0"))
    listen_port = int(settings.get("listen_port", 8080))
    environment = dict(settings.get("environment", {}))
    environment.setdefault(
        "INFERENCE_METRICS_URL",
        f"http://{config.name}:{config.port}{inference_metrics_path}",
    )
    environment.setdefault(
        "DCGM_EXPORTER_URL",
        f"http://dcgm-exporter:9400{dcgm_metrics_path}",
    )
    environment.setdefault("MODELDOCK_DEPLOYMENT_NAME", config.name)
    environment.setdefault("MODELDOCK_ENGINE", config.engine)
    environment.setdefault("COLLECTOR_HTTP_HOST", listen_host)
    environment.setdefault("COLLECTOR_HTTP_PORT", str(listen_port))
    upstream_http_url = settings.get("upstream_http_url")
    if upstream_http_url:
        environment.setdefault("UPSTREAM_HTTP_URL", str(upstream_http_url))

    depends_on = [config.name]
    if config.dcgm_exporter.get("enabled"):
        depends_on.append("dcgm-exporter")
    extra_depends_on = settings.get("depends_on", [])
    if extra_depends_on:
        depends_on.extend(str(item) for item in extra_depends_on)

    service: dict[str, object] = {
        "image": image,
        "container_name": settings.get("container_name", f"{config.name}-{service_name}"),
        "restart": settings.get("restart", "unless-stopped"),
        "depends_on": depends_on,
        "environment": environment,
        "expose": [str(listen_port)],
    }

    publish_port = settings.get("host_port")
    if publish_port is not None and "ports" not in settings:
        service["ports"] = [f"{publish_port}:{listen_port}"]

    for key in ("ports", "volumes", "command", "entrypoint", "labels", "gpus", "healthcheck"):
        if key in settings:
            service[key] = settings[key]

    return service_name, service


def render_extra_services(config: DeploymentConfig) -> str:
    blocks: list[str] = []

    dcgm_service = _build_dcgm_exporter_service(config)
    if dcgm_service is not None:
        blocks.append(_render_service_block("dcgm-exporter", dcgm_service))

    miner_client = _build_miner_client_service(config)
    if miner_client is not None:
        service_name, service = miner_client
        blocks.append(_render_service_block(service_name, service))

    collector_service = _build_metrics_collector_service(config)
    if collector_service is not None:
        service_name, service = collector_service
        blocks.append(_render_service_block(service_name, service))

    for service_name, service in config.extra_services.items():
        if not isinstance(service, dict):
            raise ValueError(f"extra_services.{service_name} must be a mapping")
        blocks.append(_render_service_block(service_name, service))

    return "\n\n".join(blocks)


def render_compose(config: DeploymentConfig) -> str:
    template_name = f"{config.engine}_compose.yaml.j2"
    template_text = (
        resources.files("miner_cli")
        .joinpath("templates", template_name)
        .read_text(encoding="utf-8")
    )
    template = Template(template_text, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    return template.render(
        name=config.name,
        image=config.image or default_image(config.engine),
        port=config.port,
        hf_token_env=config.hf_token_env,
        hf_cache=config.hf_cache,
        shm_size=config.shm_size,
        gpu_ids=config.gpu_ids,
        env=config.env,
        launch_command=build_launch_command(config),
        launch_args=build_launch_args(config),
        extra_services_yaml=render_extra_services(config),
    )


def write_deployment_files(
    config: DeploymentConfig, source_config_path: Path | None = None
) -> DeploymentPaths:
    paths = deployment_paths(config.name)
    paths.root.mkdir(parents=True, exist_ok=True)

    compose = render_compose(config)
    paths.compose_path.write_text(compose, encoding="utf-8")

    hf_token = os.getenv(config.hf_token_env, "")
    env_lines = [
        f"{config.hf_token_env}={hf_token}",
    ]
    paths.env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    if source_config_path:
        shutil.copy2(source_config_path, paths.config_path)

    return paths


def run_compose(
    paths: DeploymentPaths, *args: str, capture_output: bool = False
) -> subprocess.CompletedProcess:
    command = [
        "docker",
        "compose",
        "-f",
        str(paths.compose_path),
        "--env-file",
        str(paths.env_path),
        *args,
    ]
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def service_url(config: DeploymentConfig) -> str:
    return f"http://127.0.0.1:{config.port}"


def wait_for_ready(
    config: DeploymentConfig, timeout: int = 900, progress: ProgressLogger | None = None
) -> None:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    last_log_at = 0.0
    url = f"{service_url(config)}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if progress is not None and time.time() - last_log_at >= 10:
                progress(f"Waiting for readiness: {url}")
                last_log_at = time.time()
            time.sleep(2)
    raise TimeoutError(f"Service did not become ready within {timeout}s: {url}")


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0
