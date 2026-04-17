from __future__ import annotations

from pathlib import Path

import yaml

from miner_cli.config import DeploymentConfig, load_config, write_template_config
from miner_cli.deploy import render_compose, write_deployment_files


def test_write_template_config_generates_engine_specific_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "demo.yaml"

    write_template_config(
        config_path,
        name="demo",
        engine="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel=2,
        port=9000,
    )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["name"] == "demo"
    assert raw["image"] == "vllm/vllm-openai:latest"
    assert raw["extra_args"] == ["--max-num-seqs", "16"]
    assert raw["port"] == 9000
    assert raw["tensor_parallel"] == 2


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    try:
        load_config(config_path)
    except ValueError as exc:
        assert "must contain a YAML mapping" in str(exc)
    else:
        raise AssertionError("load_config should reject non-mapping YAML")


def test_render_compose_embeds_launch_command_and_extra_services() -> None:
    config = DeploymentConfig(
        name="demo",
        engine="sglang",
        model="Qwen/Qwen2.5-7B-Instruct",
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        extra_args=["--tool-call-parser", "hermes"],
        extra_services={"redis": {"image": "redis:7"}},
    )

    rendered = render_compose(config)

    assert "services:" in rendered
    assert "redis:" in rendered
    assert "HF_HUB_ENABLE_HF_TRANSFER=1" in rendered
    assert "--tool-call-parser" in rendered
    assert "hermes" in rendered


def test_write_deployment_files_writes_compose_env_and_copies_source(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("miner_cli.deploy.DEFAULT_DEPLOYMENTS_DIR", tmp_path / "deployments")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    source_path = tmp_path / "source.yaml"
    source_path.write_text("name: demo\n", encoding="utf-8")

    config = DeploymentConfig(
        name="demo",
        engine="vllm",
        model="Qwen/Qwen2.5-7B-Instruct",
    )

    paths = write_deployment_files(config, source_config_path=source_path)

    assert paths.compose_path.exists()
    assert "vllm.entrypoints.openai.api_server" in paths.compose_path.read_text(encoding="utf-8")
    assert paths.env_path.read_text(encoding="utf-8") == "HF_TOKEN=secret-token\n"
    assert paths.config_path.read_text(encoding="utf-8") == "name: demo\n"
