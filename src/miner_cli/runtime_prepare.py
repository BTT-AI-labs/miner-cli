from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import DeploymentConfig, image_uses_floating_latest, load_config
from .deploy import default_image
from .doctor import config_checks, gpu_container_smoke_test
from .preparation import CheckResult, ProgressLogger, run_logged_command

SUPPORTED_RUNTIME_ENGINES = {"vllm"}


def _exception_detail(exc: OSError, command: list[str] | None = None) -> str:
    if isinstance(exc, FileNotFoundError):
        missing_command = exc.filename or (command[0] if command else "command")
        return f"{missing_command} not found"
    return str(exc)


def _summarize_process_output(result: subprocess.CompletedProcess[str]) -> str:
    lines = [
        line.strip()
        for line in ((result.stdout or "").splitlines() + (result.stderr or "").splitlines())
        if line.strip()
    ]
    if not lines:
        return f"exit={result.returncode}"

    priority_markers = (
        "permission denied",
        "cannot connect",
        "is the docker daemon running",
        "error",
        "failed",
        "denied",
        "not found",
        "timeout",
    )
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in priority_markers):
            return line

    return lines[-1]


def _run_check_command(
    label: str,
    command: list[str],
    progress: ProgressLogger | None = None,
) -> CheckResult:
    try:
        result = run_logged_command(command, progress=progress)
    except FileNotFoundError as exc:
        return CheckResult(label, "fail", _exception_detail(exc, command))
    except OSError as exc:
        return CheckResult(label, "fail", _exception_detail(exc, command))

    return CheckResult(
        label,
        "ok" if result.returncode == 0 else "fail",
        _summarize_process_output(result),
    )


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
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return CheckResult("runtime cache path", "fail", f"{cache_path} is not writable: {exc}")
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
    return _run_check_command("runtime image pull", ["docker", "pull", image], progress=progress)


def _vllm_smoke_test(image: str, progress: ProgressLogger | None = None) -> CheckResult:
    return _run_check_command(
        "runtime smoke test",
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            image,
            "--help",
        ],
        progress=progress,
    )


def engine_container_smoke_test(
    engine: str,
    image: str,
    progress: ProgressLogger | None = None,
) -> CheckResult:
    if engine == "vllm":
        return _run_check_command(
            "engine container smoke test",
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                image,
                "--help",
            ],
            progress=progress,
        )

    return CheckResult("engine container smoke test", "warn", f"no smoke test for engine: {engine}")


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
    try:
        config = resolve_runtime_config(engine, config_file)
    except Exception as exc:
        return [CheckResult("runtime config", "fail", str(exc))]

    checks = [CheckResult("runtime engine", "ok", engine)]
    if progress is not None:
        progress("Running runtime preflight checks...")
    try:
        checks.extend(config_checks(config))
    except Exception as exc:
        checks.append(CheckResult("runtime preflight checks", "fail", str(exc)))
    if engine == "vllm" and image_uses_floating_latest(config.image or default_image(engine)):
        checks.append(
            CheckResult(
                "runtime image policy",
                "warn",
                "floating latest image may drift and require newer NVIDIA drivers; pin image: for reproducible deploys",
            )
        )
    checks.append(_ensure_cache_path(Path(config.hf_cache)))
    checks.append(_hf_token_check(config.hf_token_env, require_hf_token))

    if pull:
        if progress is not None:
            progress(f"Pulling runtime image: {config.image or default_image(engine)}")
        checks.append(_pull_image(config.image or default_image(engine), progress=progress))

    if smoke_test:
        if progress is not None:
            progress("Running GPU container smoke test...")
        try:
            checks.append(gpu_container_smoke_test(progress=progress))
        except OSError as exc:
            checks.append(CheckResult("gpu container smoke test", "fail", _exception_detail(exc)))
        if progress is not None:
            progress("Running engine container startup smoke test...")
        checks.append(engine_container_smoke_test(engine, config.image or default_image(engine), progress=progress))
        if progress is not None:
            progress("Running vLLM import smoke test...")
        checks.append(_vllm_smoke_test(config.image or default_image(engine), progress=progress))

    return checks
