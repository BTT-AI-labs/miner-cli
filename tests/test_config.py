from __future__ import annotations

import pytest

from miner_cli.config import DeploymentConfig, default_image_for_engine, image_uses_floating_latest


def test_from_dict_normalizes_and_validates_string_fields() -> None:
    config = DeploymentConfig.from_dict(
        {
            "name": "demo",
            "engine": "sglang",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "extra_args": ["--tool-call-parser", "hermes"],
            "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
            "dcgm_exporter": {"enabled": True},
            "miner_client": {"enabled": True, "image": "example/miner-client:latest"},
            "extra_services": {"redis": {"image": "redis:7"}},
        }
    )

    assert config.extra_args == ["--tool-call-parser", "hermes"]
    assert config.env["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert config.miner_client["image"] == "example/miner-client:latest"
    assert config.extra_services["redis"]["image"] == "redis:7"


def test_from_dict_accepts_custom_service_as_backward_compatible_alias() -> None:
    config = DeploymentConfig.from_dict(
        {
            "name": "demo",
            "engine": "sglang",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "custom_service": {"enabled": True, "image": "example/miner-client:latest"},
        }
    )

    assert config.miner_client["image"] == "example/miner-client:latest"


def test_from_dict_rejects_both_miner_client_and_custom_service() -> None:
    with pytest.raises(ValueError, match="Use only one of miner_client or custom_service"):
        DeploymentConfig.from_dict(
            {
                "name": "demo",
                "engine": "sglang",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "miner_client": {"enabled": True, "image": "example/miner-client:latest"},
                "custom_service": {"enabled": True, "image": "example/custom-service:latest"},
            }
        )


@pytest.mark.parametrize(
    ("field_name", "payload", "message"),
    [
        ("extra_args", {"extra_args": "--bad"}, "extra_args must be a list of strings"),
        ("env", {"env": {"OK": 1}}, "env.OK must be a string"),
        (
            "extra_services",
            {"extra_services": {"redis": "redis:7"}},
            "extra_services.redis must be a mapping",
        ),
    ],
)
def test_from_dict_rejects_invalid_nested_types(
    field_name: str, payload: dict[str, object], message: str
) -> None:
    raw = {
        "name": "demo",
        "engine": "sglang",
        "model": "Qwen/Qwen2.5-7B-Instruct",
    }
    raw.update(payload)

    with pytest.raises(ValueError, match=message):
        DeploymentConfig.from_dict(raw)


def test_validate_rejects_empty_runtime_fields() -> None:
    config = DeploymentConfig(name="demo", engine="vllm", model="m", host="")

    with pytest.raises(ValueError, match="host is required"):
        config.validate()


def test_default_image_for_engine() -> None:
    assert default_image_for_engine("sglang") == "lmsysorg/sglang:latest"
    assert default_image_for_engine("vllm") == "vllm/vllm-openai:latest"
    assert default_image_for_engine("vllm", image_policy="latest") == "vllm/vllm-openai:latest"

    with pytest.raises(ValueError, match="Unsupported engine"):
        default_image_for_engine("unknown")
    with pytest.raises(ValueError, match="Unsupported image policy"):
        default_image_for_engine("vllm", image_policy="preview")


def test_image_uses_floating_latest() -> None:
    assert image_uses_floating_latest("vllm/vllm-openai:latest") is True
    assert image_uses_floating_latest("vllm/vllm-openai") is True
    assert image_uses_floating_latest("vllm/vllm-openai:v0.9.0") is False
