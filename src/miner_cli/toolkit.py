from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from getpass import getuser
from importlib import resources
from pathlib import Path

from .preparation import CheckResult, ProgressLogger, run_logged_command


@dataclass(frozen=True)
class HostProfile:
    identifier: str
    family: str
    distro: str
    version: str
    arch: str
    supported: bool
    reason: str = ""


@dataclass(frozen=True)
class ToolkitComponent:
    identifier: str
    label: str


SUPPORTED_COMPONENTS = (
    ToolkitComponent("docker", "docker engine"),
    ToolkitComponent("docker-user-access", "docker user access"),
    ToolkitComponent("nvidia-container-toolkit", "nvidia container toolkit"),
    ToolkitComponent("docker-nvidia-runtime", "docker nvidia runtime"),
)

DISTRO_FAMILY_MAP = {
    "ubuntu": "debian",
    "debian": "debian",
    "linuxmint": "debian",
    "pop": "debian",
    "rhel": "rhel",
    "rocky": "rhel",
    "almalinux": "rhel",
    "centos": "rhel",
    "fedora": "rhel",
    "amzn": "rhel",
    "arch": "arch",
    "manjaro": "arch",
    "endeavouros": "arch",
}

COMPONENT_SCRIPT_MAP = {
    "docker": {
        "debian": "install_docker_linux.sh",
        "rhel": "install_docker_linux.sh",
        "arch": "install_docker_arch.sh",
    },
    "docker-user-access": {
        "debian": "ensure_docker_user_group.sh",
        "rhel": "ensure_docker_user_group.sh",
        "arch": "ensure_docker_user_group.sh",
    },
    "nvidia-container-toolkit": {
        "debian": "install_nvidia_container_toolkit_debian.sh",
        "rhel": "install_nvidia_container_toolkit_rhel.sh",
        "arch": "install_nvidia_container_toolkit_arch.sh",
    },
    "docker-nvidia-runtime": {
        "debian": "configure_docker_nvidia_runtime.sh",
        "rhel": "configure_docker_nvidia_runtime.sh",
        "arch": "configure_docker_nvidia_runtime.sh",
    },
}


def _read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as infile:
            for line in infile:
                if "=" not in line:
                    continue
                key, value = line.rstrip().split("=", 1)
                values[key] = value.strip('"')
    except OSError:
        return values
    return values


def detect_host_profile(os_release_path: Path = Path("/etc/os-release")) -> HostProfile:
    system = platform.system().lower()
    arch = platform.machine().lower()
    if system != "linux":
        return HostProfile(
            "unsupported",
            "unknown",
            "unknown",
            "unknown",
            arch,
            False,
            f"unsupported OS: {system}",
        )
    if arch not in {"x86_64", "amd64"}:
        return HostProfile(
            "unsupported",
            "unknown",
            "unknown",
            "unknown",
            arch,
            False,
            f"unsupported arch: {arch}",
        )

    os_release = _read_os_release(os_release_path)
    distro = os_release.get("ID", "unknown")
    version = os_release.get("VERSION_ID", "unknown")
    family = DISTRO_FAMILY_MAP.get(distro, "unknown")
    if family != "unknown":
        return HostProfile(f"{family}-{distro}-{version}", family, distro, version, arch, True)
    reason = f"unsupported distro family: {distro} {version}"
    return HostProfile("unsupported", family, distro, version, arch, False, reason)


def verify_toolkit_component(component_id: str) -> list[CheckResult]:
    from .doctor import docker_checks, docker_runtime_checks, gpu_checks

    if component_id == "docker":
        checks = docker_checks()
        return [check for check in checks if check.label in {"docker cli", "docker compose", "docker daemon"}]
    if component_id == "docker-user-access":
        checks = docker_checks()
        return [check for check in checks if check.label == "docker permissions"]
    if component_id == "nvidia-container-toolkit":
        checks = docker_checks() + gpu_checks()
        return [
            check
            for check in checks
            if check.label in {"nvidia container toolkit", "nvidia-smi", "gpu inventory"}
        ]
    if component_id == "docker-nvidia-runtime":
        return docker_runtime_checks()
    raise ValueError(f"Unknown toolkit component: {component_id}")


def verify_toolkit_host(
    include_smoke_test: bool = False,
    progress: ProgressLogger | None = None,
) -> list[CheckResult]:
    from .doctor import gpu_container_smoke_test, host_checks

    checks = host_checks()
    if include_smoke_test:
        if progress is not None:
            progress("Running GPU container smoke test...")
        checks.append(gpu_container_smoke_test(progress=progress))
    return checks


def _script_path(script_name: str) -> Path:
    return Path(resources.files("miner_cli").joinpath("scripts", script_name))


def run_installer_script(
    script_name: str,
    progress: ProgressLogger | None = None,
) -> subprocess.CompletedProcess[str]:
    script_path = _script_path(script_name)
    return run_logged_command(
        ["bash", str(script_path)],
        progress=progress,
        env={"PATH": str(Path("/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")), "USER": getuser()},
    )


def _summarize_process_output(completed: subprocess.CompletedProcess[str]) -> str:
    lines = [
        line.strip()
        for line in (completed.stdout.splitlines() + completed.stderr.splitlines())
        if line.strip()
    ]
    if not lines:
        return f"exit={completed.returncode}"

    for line in reversed(lines):
        lower_line = line.lower()
        if line.startswith(("E:", "Err:", "error", "ERROR")) or "failed" in lower_line:
            return line

    return lines[-1]


def manual_install_instructions(profile: HostProfile) -> str:
    if profile.family == "debian":
        return "Install Docker and NVIDIA Container Toolkit with your distro package manager, then rerun `miner-cli toolkit verify`."
    if profile.family == "rhel":
        return "Install Docker and NVIDIA Container Toolkit with dnf/yum, then rerun `miner-cli toolkit verify`."
    if profile.family == "arch":
        return "Install `docker` and `nvidia-container-toolkit` with pacman, enable the docker service, then rerun `miner-cli toolkit verify`."
    return "Run `miner-cli toolkit verify` for checks and install Docker plus NVIDIA Container Toolkit manually for this distro."


def installer_script_for(component_id: str, profile: HostProfile) -> str | None:
    return COMPONENT_SCRIPT_MAP.get(component_id, {}).get(profile.family)


def _component_is_ready(component_id: str, checks: list[CheckResult]) -> bool:
    if not checks:
        return False

    if component_id == "docker-user-access":
        return all(check.status == "ok" for check in checks)

    if component_id == "docker-nvidia-runtime":
        return all(check.status == "ok" for check in checks)

    return all(check.status != "fail" for check in checks)


def _post_install_guidance(progress: ProgressLogger | None = None) -> list[CheckResult]:
    checks = verify_toolkit_host(include_smoke_test=False, progress=progress)
    docker_permissions = next((check for check in checks if check.label == "docker permissions"), None)
    docker_daemon = next((check for check in checks if check.label == "docker daemon"), None)
    results: list[CheckResult] = []

    if docker_permissions and docker_permissions.status != "ok":
        results.append(
            CheckResult(
                "session refresh",
                "warn",
                "docker group membership may need a new shell; run `newgrp docker` or open a new shell session",
            )
        )
    if docker_daemon and docker_daemon.status != "ok":
        results.append(
            CheckResult(
                "post-install verify",
                "warn",
                f"docker daemon still not ready for the current session: {docker_daemon.detail}",
            )
        )
    if not results:
        results.append(CheckResult("post-install verify", "ok", "host prerequisites look ready"))
    return results


def install_toolkit(progress: ProgressLogger | None = None) -> list[CheckResult]:
    profile = detect_host_profile()
    if not profile.supported:
        return [
            CheckResult("host profile", "fail", profile.reason or "unsupported host"),
            CheckResult("manual install", "warn", manual_install_instructions(profile)),
        ]

    results = [CheckResult("host profile", "ok", profile.identifier)]
    for component in SUPPORTED_COMPONENTS:
        component_checks = verify_toolkit_component(component.identifier)
        if _component_is_ready(component.identifier, component_checks):
            results.append(CheckResult(component.label, "ok", "already ready"))
            continue

        script_name = installer_script_for(component.identifier, profile)
        if script_name is None:
            results.append(
                CheckResult(
                    component.label,
                    "warn",
                    f"no installer backend for {profile.family}; {manual_install_instructions(profile)}",
                )
            )
            break

        if progress is not None:
            progress(f"Installing {component.label} with backend {script_name}...")
        completed = run_installer_script(script_name, progress=progress)
        status = "ok" if completed.returncode == 0 else "fail"
        results.append(
            CheckResult(
                component.label,
                status,
                _summarize_process_output(completed),
            )
        )
        if completed.returncode != 0:
            break

    if not any(check.status == "fail" for check in results):
        if progress is not None:
            progress("Running post-install verification...")
        results.extend(_post_install_guidance(progress=progress))

    return results
