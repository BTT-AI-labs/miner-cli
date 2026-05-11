from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_DEPLOYMENTS_DIR = Path.home() / ".miner-cli" / "deployments"
SUPPORTED_ENGINES = frozenset({"sglang", "vllm"})
SUPPORTED_IMAGE_POLICIES = frozenset({"stable", "latest"})
STABLE_ENGINE_IMAGES = {
    "sglang": "lmsysorg/sglang:latest",
    "vllm": "vllm/vllm-openai:latest",
}
LATEST_ENGINE_IMAGES = {
    "sglang": "lmsysorg/sglang:latest",
    "vllm": "vllm/vllm-openai:latest",
}


def default_image_for_engine(engine: str, image_policy: str = "stable") -> str:
    if image_policy not in SUPPORTED_IMAGE_POLICIES:
        raise ValueError(
            f"Unsupported image policy: {image_policy}. Expected one of: stable, latest"
        )
    images = STABLE_ENGINE_IMAGES if image_policy == "stable" else LATEST_ENGINE_IMAGES
    try:
        return images[engine]
    except KeyError as exc:
        raise ValueError(f"Unsupported engine: {engine}") from exc


def image_uses_floating_latest(image: str) -> bool:
    normalized = image.strip()
    return normalized.endswith(":latest") or ":" not in normalized


def _ensure_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _ensure_string_map(value: Any, field_name: str) -> dict[str, str]:
    mapping = _ensure_mapping(value, field_name)
    normalized: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        if not isinstance(item, str):
            raise ValueError(f"{field_name}.{key} must be a string")
        normalized[key] = item
    return normalized


def _ensure_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return list(value)


@dataclass
class DeploymentConfig:
    name: str
    engine: str
    model: str
    image: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel: int = 1
    gpu_ids: str = "all"
    trust_remote_code: bool = True
    dtype: str | None = "bfloat16"
    max_model_len: int | None = None
    api_key: str | None = None
    hf_token_env: str = "HF_TOKEN"
    hf_cache: str = "/data/huggingface"
    shm_size: str = "16g"
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    dcgm_exporter: dict[str, Any] = field(default_factory=dict)
    miner_client: dict[str, Any] = field(default_factory=dict)
    extra_services: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DeploymentConfig":
        env = _ensure_string_map(raw.get("env"), "env")
        extra_args = _ensure_string_list(raw.get("extra_args"), "extra_args")
        dcgm_exporter = _ensure_mapping(raw.get("dcgm_exporter"), "dcgm_exporter")
        miner_client = _load_miner_client_config(raw)
        extra_services = _ensure_mapping(raw.get("extra_services"), "extra_services")
        cfg = cls(
            name=raw["name"],
            engine=raw["engine"],
            model=raw["model"],
            image=raw.get("image"),
            host=raw.get("host", "0.0.0.0"),
            port=int(raw.get("port", 8000)),
            tensor_parallel=int(raw.get("tensor_parallel", 1)),
            gpu_ids=str(raw.get("gpu_ids", "all")),
            trust_remote_code=bool(raw.get("trust_remote_code", True)),
            dtype=raw.get("dtype", "bfloat16"),
            max_model_len=raw.get("max_model_len"),
            api_key=raw.get("api_key"),
            hf_token_env=raw.get("hf_token_env", "HF_TOKEN"),
            hf_cache=raw.get("hf_cache", "/data/huggingface"),
            shm_size=raw.get("shm_size", "16g"),
            extra_args=extra_args,
            env=env,
            dcgm_exporter=dcgm_exporter,
            miner_client=miner_client,
            extra_services=extra_services,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.name:
            raise ValueError("name is required")
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError("engine must be one of: sglang, vllm")
        if not self.model:
            raise ValueError("model is required")
        if self.image is not None and not self.image.strip():
            raise ValueError("image must not be empty when set")
        if not self.host:
            raise ValueError("host is required")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.tensor_parallel <= 0:
            raise ValueError("tensor_parallel must be >= 1")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("max_model_len must be >= 1 when set")
        if not self.hf_token_env:
            raise ValueError("hf_token_env is required")
        if not self.hf_cache:
            raise ValueError("hf_cache is required")
        if not self.shm_size:
            raise ValueError("shm_size is required")
        _ensure_string_list(self.extra_args, "extra_args")
        _ensure_string_map(self.env, "env")
        _ensure_mapping(self.dcgm_exporter, "dcgm_exporter")
        _ensure_mapping(self.miner_client, "miner_client")
        extra_services = _ensure_mapping(self.extra_services, "extra_services")
        for service_name, service in extra_services.items():
            if not isinstance(service_name, str):
                raise ValueError("extra_services keys must be strings")
            if not isinstance(service, Mapping):
                raise ValueError(f"extra_services.{service_name} must be a mapping")


def load_config(path: Path) -> DeploymentConfig:
    with path.open("r", encoding="utf-8") as infile:
        raw = yaml.safe_load(infile) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return DeploymentConfig.from_dict(raw)


def write_template_config(
    path: Path,
    name: str,
    engine: str,
    model: str,
    tensor_parallel: int = 1,
    port: int = 8000,
    image: str | None = None,
    image_policy: str = "stable",
) -> None:
    image = default_image_for_engine(engine, image_policy=image_policy) if image is None else image
    template = {
        "name": name,
        "engine": engine,
        "model": model,
        "image": image,
        "host": "0.0.0.0",
        "port": port,
        "tensor_parallel": tensor_parallel,
        "gpu_ids": "all",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "hf_token_env": "HF_TOKEN",
        "hf_cache": "/data/huggingface",
        "shm_size": "16g",
        "extra_args": [
            "--tool-call-parser hermes",
        ]
        if engine == "sglang"
        else [
            "--max-num-seqs",
            "16",
        ],
        "env": {},
        "dcgm_exporter": {
            "enabled": False,
            "gpus": "all"
        },
        "miner_client": {
            "enabled": False,
            "image": "lzwukeyou/miner-agent:0.0.1",
            "listen_host": "127.0.0.1",
            "listen_port": 8080,
            "public_ip": "${your public ip}",
            "gpus": "all",
            "environment": {
                "LOG_LEVEL": "info",
                "MAIN_API_BASE_URL": "${ask for dev}",
                "MINER_TOKEN": "${replace-me}",
                "MINER_TARGET_MODEL": model,
                "MINER_HOME": "/root/.miner",
                "MINER_RUNTIME_TYPE": engine,
            },
        },
        "extra_services": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        yaml.safe_dump(template, outfile, sort_keys=False)


def _load_miner_client_config(raw: dict[str, Any]) -> dict[str, Any]:
    miner_client = raw.get("miner_client")
    custom_service = raw.get("custom_service")
    if miner_client is not None and custom_service is not None:
        raise ValueError("Use only one of miner_client or custom_service")
    if miner_client is not None:
        return _ensure_mapping(miner_client, "miner_client")
    if custom_service is not None:
        return _ensure_mapping(custom_service, "custom_service")
    return {}
