from __future__ import annotations

import getpass
import grp
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path

from .config import DeploymentConfig
from .deploy import default_image, is_port_in_use
from .preparation import CheckResult, ProgressLogger, run_logged_command


def _summarize_command_output(result: subprocess.CompletedProcess[str]) -> str:
    lines = [
        line.strip()
        for line in (result.stdout.splitlines() + result.stderr.splitlines())
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
    )
    for line in reversed(lines):
        lower_line = line.lower()
        if any(marker in lower_line for marker in priority_markers):
            return line

    return lines[-1]


def _run(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return False, "not found"
    if result.returncode == 0:
        detail = _summarize_command_output(result)
        return True, detail if detail else "ok"
    return False, _summarize_command_output(result)


def _linux_os_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    system = platform.system().lower()
    if system != "linux":
        results.append(CheckResult("operating system", "fail", f"unsupported system: {system}"))
        return results

    results.append(CheckResult("operating system", "ok", "linux"))
    arch = platform.machine().lower()
    if arch not in {"x86_64", "amd64"}:
        results.append(CheckResult("architecture", "fail", f"unsupported arch: {arch}"))
    else:
        results.append(CheckResult("architecture", "ok", arch))

    os_release = {}
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as infile:
            for line in infile:
                if "=" not in line:
                    continue
                key, value = line.rstrip().split("=", 1)
                os_release[key] = value.strip('"')
    except OSError as exc:
        results.append(
            CheckResult("distribution", "warn", f"unable to read /etc/os-release: {exc}")
        )
        return results

    distro = os_release.get("ID", "unknown")
    version = os_release.get("VERSION_ID", "unknown")
    if distro == "ubuntu" and version in {"22.04", "24.04"}:
        results.append(CheckResult("distribution", "ok", f"{distro} {version}"))
    elif distro == "ubuntu":
        results.append(CheckResult("distribution", "warn", f"{distro} {version} (untested)"))
    else:
        results.append(
            CheckResult("distribution", "warn", f"{distro} {version} (tool is tuned for Ubuntu)")
        )

    results.append(CheckResult("kernel", "ok", platform.release()))
    return results


def docker_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    ok, detail = _run(["docker", "--version"])
    results.append(CheckResult("docker cli", "ok" if ok else "fail", detail))

    ok, detail = _run(["docker", "compose", "version"])
    results.append(CheckResult("docker compose", "ok" if ok else "fail", detail))

    ok, detail = _run(["docker", "info"])
    results.append(CheckResult("docker daemon", "ok" if ok else "fail", detail))

    user = getpass.getuser()
    in_docker_group = False
    try:
        docker_group = grp.getgrnam("docker")
        in_docker_group = user in docker_group.gr_mem or os.getgid() == docker_group.gr_gid
    except KeyError:
        pass
    if in_docker_group or os.geteuid() == 0:
        results.append(CheckResult("docker permissions", "ok", f"user={user}"))
    else:
        results.append(
            CheckResult(
                "docker permissions",
                "warn",
                f"user={user} may need sudo or docker group membership",
            )
        )

    toolkit_present = any(
        Path(path).exists() for path in ("/usr/bin/nvidia-container-runtime", "/usr/bin/nvidia-ctk")
    )
    results.append(
        CheckResult(
            "nvidia container toolkit",
            "ok" if toolkit_present else "fail",
            "present" if toolkit_present else "missing",
        )
    )

    return results


def docker_runtime_checks() -> list[CheckResult]:
    ok, detail = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
    if not ok:
        return [CheckResult("docker nvidia runtime", "warn", detail)]
    if detail == "null":
        return [
            CheckResult(
                "docker nvidia runtime",
                "warn",
                "docker daemon unavailable or current user cannot inspect runtimes",
            )
        ]
    try:
        runtimes = json.loads(detail)
    except json.JSONDecodeError:
        return [CheckResult("docker nvidia runtime", "warn", "unable to parse docker runtimes")]
    configured = isinstance(runtimes, dict) and "nvidia" in runtimes
    return [
        CheckResult(
            "docker nvidia runtime",
            "ok" if configured else "fail",
            "configured" if configured else "not configured",
        )
    ]


def gpu_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    ok, detail = _run(["nvidia-smi"])
    results.append(CheckResult("nvidia-smi", "ok" if ok else "fail", detail))
    if not ok:
        return results

    ok, detail = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,mig.mode.current",
            "--format=csv,noheader",
        ]
    )
    if not ok:
        results.append(CheckResult("gpu inventory", "warn", detail))
        return results

    rows = [line.strip() for line in detail.splitlines() if line.strip()]
    results.append(CheckResult("gpu count", "ok", str(len(rows))))
    if not rows:
        results.append(CheckResult("gpu inventory", "fail", "no GPUs detected"))
        return results

    names = []
    mig_enabled = False
    min_mem = None
    driver_version = None
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 4:
            names.append(parts[0])
            mem = parts[1]
            driver_version = driver_version or parts[2]
            mig_enabled = mig_enabled or parts[3].lower() == "enabled"
            try:
                mem_value = int(mem.split()[0])
                min_mem = mem_value if min_mem is None else min(min_mem, mem_value)
            except (ValueError, IndexError):
                pass
    results.append(CheckResult("gpu model", "ok", ", ".join(sorted(set(names)))))
    if driver_version:
        results.append(CheckResult("nvidia driver", "ok", driver_version))
    if min_mem is not None:
        results.append(CheckResult("min gpu memory", "ok", f"{min_mem} MiB"))
    results.append(
        CheckResult(
            "mig mode", "warn" if mig_enabled else "ok", "enabled" if mig_enabled else "disabled"
        )
    )
    return results


def _storage_and_network_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (1024**3)
    status = "ok" if free_gb >= 200 else "warn" if free_gb >= 100 else "fail"
    results.append(CheckResult("root disk free", status, f"{free_gb} GiB"))

    shm_total = shutil.disk_usage("/dev/shm").total // (1024**3) if Path("/dev/shm").exists() else 0
    shm_status = "ok" if shm_total >= 8 else "warn"
    results.append(CheckResult("/dev/shm size", shm_status, f"{shm_total} GiB"))

    try:
        socket.gethostbyname("huggingface.co")
        results.append(CheckResult("dns:huggingface.co", "ok", "resolved"))
    except OSError as exc:
        results.append(CheckResult("dns:huggingface.co", "warn", str(exc)))

    return results


def host_checks() -> list[CheckResult]:
    return [
        *_linux_os_checks(),
        *docker_checks(),
        *docker_runtime_checks(),
        *gpu_checks(),
        *_storage_and_network_checks(),
    ]


def _inspect_image_availability(image: str) -> CheckResult:
    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", image],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return CheckResult(
            "image availability", "warn", f"unable to inspect {image}: docker not found"
        )

    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        return CheckResult(
            "image availability",
            "warn",
            f"unable to inspect {image}: {output or f'exit={result.returncode}'}",
        )

    try:
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError:
        summary = output.splitlines()[0] if output else "available"
        return CheckResult("image availability", "ok", summary)

    if isinstance(manifest, dict):
        manifests = manifest.get("manifests")
        if isinstance(manifests, list) and manifests:
            platforms: list[str] = []
            for item in manifests:
                platform_info = item.get("platform", {}) if isinstance(item, dict) else {}
                os_name = platform_info.get("os")
                arch = platform_info.get("architecture")
                if os_name and arch:
                    platform_label = f"{os_name}/{arch}"
                    variant = platform_info.get("variant")
                    if variant:
                        platform_label = f"{platform_label}/{variant}"
                    platforms.append(platform_label)
            if platforms:
                unique_platforms = sorted(set(platforms))
                return CheckResult(
                    "image availability",
                    "ok",
                    f"available ({len(unique_platforms)} platforms: {', '.join(unique_platforms)})",
                )

        digest = manifest.get("Descriptor", {}).get("digest")
        if digest:
            return CheckResult("image availability", "ok", f"available ({digest})")

    return CheckResult("image availability", "ok", "available")


def config_checks(config: DeploymentConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    if is_port_in_use(config.port):
        results.append(
            CheckResult("configured port", "fail", f"port {config.port} is already in use")
        )
    else:
        results.append(CheckResult("configured port", "ok", f"port {config.port} is available"))

    cache_dir = Path(config.hf_cache)
    target = cache_dir if cache_dir.exists() else cache_dir.parent
    writable = os.access(target, os.W_OK)
    results.append(
        CheckResult(
            "hf cache path",
            "ok" if writable else "fail",
            f"{config.hf_cache} {'is writable' if writable else 'is not writable'}",
        )
    )

    token_present = bool(os.getenv(config.hf_token_env))
    results.append(
        CheckResult(
            "hugging face token",
            "ok" if token_present else "warn",
            f"{config.hf_token_env} {'is set' if token_present else 'is not set'}",
        )
    )

    image = config.image or default_image(config.engine)
    results.append(CheckResult("engine image", "ok", image))

    results.append(_inspect_image_availability(image))

    ok, detail = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if ok:
        gpu_count = len([line for line in detail.splitlines() if line.strip()])
        gpu_status = "ok" if gpu_count >= config.tensor_parallel else "fail"
        results.append(
            CheckResult(
                "tensor parallel fit",
                gpu_status,
                f"need {config.tensor_parallel}, found {gpu_count}",
            )
        )
    else:
        results.append(CheckResult("tensor parallel fit", "warn", "unable to query GPUs"))

    return results


def gpu_container_smoke_test(progress: ProgressLogger | None = None) -> CheckResult:
    if progress is None:
        ok, detail = _run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.4.1-base-ubuntu22.04",
                "nvidia-smi",
            ]
        )
        return CheckResult(
            "gpu container smoke test",
            "ok" if ok else "fail",
            detail,
        )

    result = run_logged_command(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "nvidia/cuda:12.4.1-base-ubuntu22.04",
            "nvidia-smi",
        ],
        progress=progress,
    )
    detail = _summarize_command_output(result)
    return CheckResult(
        "gpu container smoke test",
        "ok" if result.returncode == 0 else "fail",
        detail,
    )
