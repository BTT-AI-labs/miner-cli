from __future__ import annotations

import os
from pathlib import Path

from .config import DeploymentConfig, load_config
from .deploy import default_image
from .doctor import config_checks, gpu_container_smoke_test
from .preparation import CheckResult, ProgressLogger, run_logged_command

SUPPORTED_RUNTIME_ENGINES = {"vllm"}


def resolve_runtime_config(
    engine: str,
    config_file: Path | None = None,
) -> DeploymentConfig:
    if config_file is not None:
        config = load_config(config_file)
        if config.engine != engine:
            raise ValueError(
                f"Config engine mismatch: expected {engine}, found {config.engine}"
            )
        return config

    return DeploymentConfig(
        name=f"{engine}-prepare",
        engine=engine,
        model="placeholder/model",
        image=default_image(engine),
        hf_cache=str(Path.home() / ".cache" / "huggingface"),
    )


def _ensure_cache_path(cache_path: Path) -> CheckResult:
    target = cache_path if cache_path.exists() else cache_path.parent
    target.mkdir(parents=True, exist_ok=True)
    if os.access(target, os.W_OK):
        return CheckResult("runtime cache path", "ok", f"{cache_path} is writable")
    return CheckResult("runtime cache path", "fail", f"{cache_path} is not writable")


def _hf_token_check(env_name: str, require_hf_token: bool) -> CheckResult:
    present = bool(os.getenv(env_name))
    if present:
        return CheckResult("hugging face token", "ok", f"{env_name} is set")
    status = "fail" if require_hf_token else "warn"
    detail = f"{env_name} is not set"
    return CheckResult("hugging face token", status, detail)


def _pull_image(image: str, progress: ProgressLogger | None = None) -> CheckResult:
    result = run_logged_command(["docker", "pull", image], progress=progress)
    detail = (result.stdout or result.stderr).strip()
    return CheckResult(
        "runtime image pull",
        "ok" if result.returncode == 0 else "fail",
        detail.splitlines()[-1] if detail else f"exit={result.returncode}",
    )


def _vllm_smoke_test(image: str, progress: ProgressLogger | None = None) -> CheckResult:
    result = run_logged_command(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "python",
            image,
            "-c",
            "import vllm; print('vllm-ready')",
        ],
        progress=progress,
    )
    detail = (result.stdout or result.stderr).strip()
    return CheckResult(
        "runtime smoke test",
        "ok" if result.returncode == 0 else "fail",
        detail.splitlines()[-1] if detail else f"exit={result.returncode}",
    )


def prepare_runtime(
    engine: str,
    config_file: Path | None = None,
    pull: bool = True,
    smoke_test: bool = False,
    require_hf_token: bool = False,
    progress: ProgressLogger | None = None,
) -> list[CheckResult]:
    if engine not in SUPPORTED_RUNTIME_ENGINES:
        return [CheckResult("runtime engine", "fail", f"unsupported engine: {engine}")]

    if progress is not None:
        progress(f"Resolving runtime config for engine: {engine}")
    config = resolve_runtime_config(engine, config_file)
    checks = [CheckResult("runtime engine", "ok", engine)]
    if progress is not None:
        progress("Running runtime preflight checks...")
    checks.extend(config_checks(config))
    checks.append(_ensure_cache_path(Path(config.hf_cache)))
    checks.append(_hf_token_check(config.hf_token_env, require_hf_token))

    if pull:
        if progress is not None:
            progress(f"Pulling runtime image: {config.image or default_image(engine)}")
        checks.append(_pull_image(config.image or default_image(engine), progress=progress))

    if smoke_test:
        if progress is not None:
            progress("Running GPU container smoke test...")
        checks.append(gpu_container_smoke_test(progress=progress))
        if progress is not None:
            progress("Running vLLM import smoke test...")
        checks.append(_vllm_smoke_test(config.image or default_image(engine), progress=progress))

    return checks
