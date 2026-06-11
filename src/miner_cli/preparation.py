from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table


@dataclass
class CheckResult:
    label: str
    status: str
    detail: str


STATUS_COLOR = {"ok": "green", "warn": "yellow", "fail": "red"}
ProgressLogger = Callable[[str], None]


def _classify_nvidia_host_failure(detail: str) -> str:
    lowered = detail.lower()
    if lowered == "not found":
        return "driver_missing"
    if "couldn't communicate with the nvidia driver" in lowered:
        return "driver_not_loaded"
    if "no devices were found" in lowered or "no devices found" in lowered:
        return "gpu_not_visible"
    if "insufficiently permission" in lowered or "permission denied" in lowered:
        return "permission_or_device_nodes"
    return "unknown"


def _classify_gpu_runtime_failure(detail: str) -> str:
    lowered = detail.lower()
    if "driver version is insufficient" in lowered:
        return "driver_too_old"
    if "unsatisfied condition: cuda>=" in lowered:
        return "driver_too_old"
    if "could not select device driver" in lowered:
        return "docker_runtime_not_wired"
    if "nvidia-container-cli" in lowered:
        return "container_runtime_or_driver"
    if "manifest for " in lowered and "not found" in lowered:
        return "image_missing"
    return "unknown"


def remediation_steps(checks: list[CheckResult]) -> list[str]:
    steps: list[str] = []
    for check in checks:
        if check.status == "ok":
            continue

        detail = check.detail.lower()
        if check.label == "operating system":
            steps.append("Use a Linux host. `miner-cli` does not support macOS or Windows as deployment hosts.")
        elif check.label == "architecture":
            steps.append("Use an x86_64 Linux host for GPU deployment.")
        elif check.label == "distribution":
            steps.append(
                "Ubuntu 22.04/24.04 is the best-supported path. On other distros, expect to install Docker and NVIDIA tooling manually."
            )
        elif check.label in {"docker cli", "docker compose"}:
            steps.append("Run `miner-cli toolkit install` to install Docker prerequisites, then rerun `miner-cli toolkit verify`.")
        elif check.label == "docker daemon":
            if "permission denied" in detail:
                steps.append("Run Docker with elevated privileges or add your user to the `docker` group, then open a new shell.")
            else:
                steps.append("Start the Docker daemon and rerun `miner-cli toolkit verify`.")
        elif check.label == "docker permissions":
            steps.append("Add your user to the `docker` group, then run `newgrp docker` or open a new shell session.")
        elif check.label == "nvidia container toolkit":
            steps.append("Run `miner-cli toolkit install` to install NVIDIA Container Toolkit, then rerun `miner-cli toolkit verify --smoke-test`.")
        elif check.label == "docker nvidia runtime":
            steps.append("Run `miner-cli toolkit install` to configure Docker's `nvidia` runtime, then rerun `miner-cli toolkit verify --smoke-test`.")
        elif check.label == "nvidia-smi":
            failure_kind = _classify_nvidia_host_failure(check.detail)
            if failure_kind == "driver_missing":
                steps.append("Install the host NVIDIA driver first, confirm `nvidia-smi` exists and runs, then rerun `miner-cli toolkit verify`.")
            elif failure_kind == "driver_not_loaded":
                steps.append("Repair the host NVIDIA driver/kernel module state, then confirm `nvidia-smi` works on the host before rerunning `miner-cli toolkit verify`.")
            elif failure_kind == "gpu_not_visible":
                steps.append("The NVIDIA driver appears present but no GPU is visible. Check PCI visibility, passthrough/cloud GPU attachment, then rerun `miner-cli toolkit verify`.")
            elif failure_kind == "permission_or_device_nodes":
                steps.append("Check access to NVIDIA device nodes such as `/dev/nvidia*`, then rerun `miner-cli toolkit verify`.")
            else:
                steps.append("Install or repair the host NVIDIA driver first, confirm `nvidia-smi` works on the host, then rerun `miner-cli toolkit verify`.")
        elif check.label == "gpu inventory":
            if "no gpus detected" in detail:
                steps.append("The driver is running but no GPU is visible to `nvidia-smi`. Check hardware visibility or VM passthrough, then rerun `miner-cli toolkit verify`.")
            else:
                steps.append("Check that the GPU is visible on the host with `nvidia-smi`, then rerun `miner-cli toolkit verify`.")
        elif check.label == "gpu container smoke test":
            failure_kind = _classify_gpu_runtime_failure(check.detail)
            if failure_kind == "driver_too_old":
                steps.append("The host NVIDIA driver is older than the CUDA requirement in the container image. Upgrade the host driver or pin an older image, then rerun `miner-cli toolkit verify --smoke-test`.")
            elif failure_kind == "docker_runtime_not_wired":
                steps.append("Docker cannot hand GPUs into the container. Run `miner-cli toolkit install` to wire the `nvidia` runtime, then rerun `miner-cli toolkit verify --smoke-test`.")
            else:
                steps.append("Run `miner-cli toolkit verify --smoke-test`. If host `nvidia-smi` works but the container test fails, repair NVIDIA Container Toolkit or Docker runtime wiring.")
        elif check.label == "configured port":
            steps.append("Choose a different `port:` in the config or stop the process already using that port, then retry.")
        elif check.label in {"hf cache path", "runtime cache path"}:
            steps.append("Point `hf_cache` to a writable directory or fix filesystem permissions before retrying.")
        elif check.label == "hugging face token":
            steps.append("Export the required Hugging Face token, for example `export HF_TOKEN=...`, if the model or runtime needs it.")
        elif check.label == "image availability":
            steps.append("Pin a valid image tag and confirm the image exists with `docker manifest inspect <image>` before retrying.")
        elif check.label in {"runtime image pull", "engine container smoke test", "runtime smoke test"}:
            failure_kind = _classify_gpu_runtime_failure(check.detail)
            if failure_kind == "driver_too_old":
                steps.append("The selected image likely needs a newer NVIDIA driver. Upgrade the host driver or pin an older image tag before retrying.")
            elif failure_kind == "image_missing":
                steps.append("The selected image tag does not exist. Pin a valid image tag before retrying.")
            else:
                steps.append(
                    "Run `miner-cli runtime prepare --engine vllm --smoke-test` to isolate image, driver, and container startup issues before `up`."
                )
        elif check.label == "deployment startup":
            steps.append("Inspect the container logs and rerun `miner-cli runtime prepare --engine vllm --smoke-test` before retrying `up`.")
        elif check.label == "service readiness":
            steps.append("Check container logs and health status, then rerun `miner-cli runtime prepare --engine vllm --smoke-test` if the model runtime still does not come up.")
        elif check.label == "tensor parallel fit":
            steps.append("Lower `tensor_parallel` or move the deployment to a host with enough GPUs.")
        elif check.label == "root disk free":
            steps.append("Free disk space on the host before pulling large model images.")
        elif check.label == "/dev/shm size":
            steps.append("Increase `/dev/shm` on the host or container runtime if model startup fails with shared-memory errors.")
        elif check.label.startswith("dns:"):
            steps.append("Fix outbound DNS/network access for model downloads and image pulls before retrying.")
        elif check.label == "manual install":
            steps.append(check.detail)
        elif check.label == "session refresh":
            steps.append(check.detail)
        elif check.label == "runtime image policy":
            steps.append("Pin `image:` to a tested tag so upstream CUDA/driver changes do not break miner startup unexpectedly.")
        elif check.label == "runtime engine":
            steps.append("Use a supported engine value such as `vllm`.")
        elif check.label == "runtime config":
            steps.append("Fix the runtime YAML config and rerun `miner-cli runtime prepare`.")
        elif check.label == "runtime preflight checks":
            steps.append("Fix the reported preflight error and rerun `miner-cli runtime prepare`.")
        elif check.label == "post-install verify":
            steps.append(check.detail)

    deduped_steps: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if step not in seen:
            deduped_steps.append(step)
            seen.add(step)
    return deduped_steps


def print_results(console: Console, title: str, checks: list[CheckResult]) -> None:
    table = Table(title=title)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in checks:
        color = STATUS_COLOR.get(check.status, "white")
        table.add_row(
            check.label,
            f"[{color}]{check.status}[/{color}]",
            check.detail,
        )
    console.print(table)
    steps = remediation_steps(checks)
    if steps:
        console.print("[bold]Next steps[/bold]")
        for step in steps:
            console.print(f"- {step}")


def run_logged_command(
    command: list[str],
    progress: ProgressLogger | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if progress is None:
        try:
            return subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                env=env,
            )
        except FileNotFoundError as exc:
            missing_command = exc.filename or command[0]
            return subprocess.CompletedProcess(
                command,
                127,
                stdout="",
                stderr=f"{missing_command} not found",
            )
        except OSError as exc:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=str(exc))

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except FileNotFoundError as exc:
        missing_command = exc.filename or command[0]
        message = f"{missing_command} not found"
        progress(message)
        return subprocess.CompletedProcess(command, 127, stdout=message, stderr="")
    except OSError as exc:
        message = str(exc)
        progress(message)
        return subprocess.CompletedProcess(command, 1, stdout=message, stderr="")

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        progress(line.rstrip())
    returncode = process.wait()
    combined_output = "".join(output_lines)
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=combined_output,
        stderr="",
    )
