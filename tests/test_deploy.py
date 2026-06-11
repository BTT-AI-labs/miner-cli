from __future__ import annotations

import pytest

from miner_cli.config import DeploymentConfig
from miner_cli.deploy import (
    _model_service_headers,
    build_launch_args,
    build_launch_command,
    render_extra_services,
)


def test_build_launch_command_for_sglang_includes_expected_flags() -> None:
    config = DeploymentConfig(
        name="demo",
        engine="sglang",
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel=4,
        max_model_len=8192,
        api_key="secret",
        extra_args=["--tool-call-parser", "hermes"],
    )

    command = build_launch_command(config)

    assert "sglang.launch_server" in command
    assert "'--tp' '4'" in command
    assert "'--context-length' '8192'" in command
    assert "'--api-key' 'secret'" in command
    assert "'--tool-call-parser' 'hermes'" in command


def test_build_launch_args_for_vllm_uses_default_entrypoint_style() -> None:
    config = DeploymentConfig(
        name="demo",
        engine="vllm",
        model="Qwen/Qwen3.5-9B",
        tensor_parallel=1,
        max_model_len=32768,
        extra_args=["--max-num-seqs", "16"],
    )

    args = build_launch_args(config)

    assert args[:3] == ["Qwen/Qwen3.5-9B", "--host", "0.0.0.0"]
    assert args[-2:] == ["--max-num-seqs", "16"]



def test_render_extra_services_includes_miner_client() -> None:
    config = DeploymentConfig(
        name="demo",
        engine="sglang",
        model="Qwen/Qwen2.5-7B-Instruct",
        dcgm_exporter={"enabled": True},
        miner_client={
            "enabled": True,
            "image": "example/miner-client:latest",
            "public_ip": "https://demo.example.com/v1/chat/completions",
            "listen_port": 7070,
            "host_port": 17070,
            "upstream_http_url": "http://internal-service:9000/api",
        },
    )

    rendered = render_extra_services(config)

    assert "miner-client:" in rendered
    assert "MODELDOCK_INFERENCE_BASE_URL: http://demo:8000" in rendered
    assert "MODELDOCK_OPENAI_BASE_URL: http://demo:8000/v1" in rendered
    assert "MODELDOCK_DCGM_EXPORTER_URL: http://dcgm-exporter:9400/metrics" in rendered
    assert "MINER_HTTP_HOST: 0.0.0.0" in rendered
    assert "MINER_HTTP_PORT: '7070'" in rendered
    assert "MINER_VLLM_BASE_URL: http://demo:8000" in rendered
    assert "MINER_DCGM_METRICS_URL: http://dcgm-exporter:9400/metrics" in rendered
    assert "UPSTREAM_HTTP_URL: http://internal-service:9000/api" in rendered
    assert "17070:7070" in rendered


def test_render_extra_services_passes_api_key_to_miner_client() -> None:
    config = DeploymentConfig(
        name="demo",
        engine="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        api_key="secret",
        miner_client={
            "enabled": True,
            "image": "example/miner-client:latest",
            "public_ip": "https://demo.example.com/v1/chat/completions",
        },
    )

    rendered = render_extra_services(config)

    assert "MINER_VLLM_API_KEY: secret" in rendered
    assert "MODELDOCK_INFERENCE_API_KEY: secret" in rendered


def test_model_service_headers_include_authorization_when_api_key_set() -> None:
    config = DeploymentConfig(
        name="demo",
        engine="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        api_key="secret",
    )

    assert _model_service_headers(config) == {"Authorization": "Bearer secret"}


def test_render_extra_services_requires_miner_client_image() -> None:
    config = DeploymentConfig(
        name="demo",
        engine="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        miner_client={"enabled": True},
    )

    with pytest.raises(ValueError, match="miner_client.image is required"):
        render_extra_services(config)
