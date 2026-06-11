from __future__ import annotations

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from miner_cli.cli import app
from miner_cli.config import DeploymentConfig
from miner_cli.preparation import CheckResult, remediation_steps
from miner_cli.runtime_prepare import (
    _ensure_cache_path,
    engine_container_smoke_test,
    prepare_runtime,
)
from miner_cli.toolkit import (
    HostProfile,
    _component_is_ready,
    _post_install_guidance,
    _summarize_process_output,
    detect_host_profile,
    install_toolkit,
    installer_script_for,
    manual_install_instructions,
    verify_toolkit_host,
)


def test_detect_host_profile_supports_ubuntu_x86(monkeypatch, tmp_path: Path) -> None:
    os_release = tmp_path / "os-release"
    os_release.write_text('ID="ubuntu"\nVERSION_ID="22.04"\n', encoding="utf-8")
    monkeypatch.setattr("miner_cli.toolkit.platform.system", lambda: "Linux")
    monkeypatch.setattr("miner_cli.toolkit.platform.machine", lambda: "x86_64")

    profile = detect_host_profile(os_release_path=os_release)

    assert profile.supported is True
    assert profile.family == "debian"
    assert profile.identifier == "debian-ubuntu-22.04"


def test_detect_host_profile_supports_rocky(monkeypatch, tmp_path: Path) -> None:
    os_release = tmp_path / "os-release"
    os_release.write_text('ID="rocky"\nVERSION_ID="9.4"\n', encoding="utf-8")
    monkeypatch.setattr("miner_cli.toolkit.platform.system", lambda: "Linux")
    monkeypatch.setattr("miner_cli.toolkit.platform.machine", lambda: "x86_64")

    profile = detect_host_profile(os_release_path=os_release)

    assert profile.supported is True
    assert profile.family == "rhel"


def test_detect_host_profile_supports_arch(monkeypatch, tmp_path: Path) -> None:
    os_release = tmp_path / "os-release"
    os_release.write_text('ID="arch"\nVERSION_ID="rolling"\n', encoding="utf-8")
    monkeypatch.setattr("miner_cli.toolkit.platform.system", lambda: "Linux")
    monkeypatch.setattr("miner_cli.toolkit.platform.machine", lambda: "x86_64")

    profile = detect_host_profile(os_release_path=os_release)

    assert profile.supported is True
    assert profile.family == "arch"


def test_install_toolkit_rejects_unsupported_host(monkeypatch) -> None:
    monkeypatch.setattr("miner_cli.toolkit.platform.system", lambda: "Darwin")

    checks = install_toolkit()

    assert len(checks) == 2
    assert checks[0].status == "fail"
    assert "unsupported" in checks[0].detail
    assert checks[1].label == "manual install"


def test_installer_script_for_debian_profile() -> None:
    profile = HostProfile("debian-ubuntu-24.04", "debian", "ubuntu", "24.04", "x86_64", True)

    assert installer_script_for("docker", profile) == "install_docker_linux.sh"
    assert installer_script_for("docker-user-access", profile) == "ensure_docker_user_group.sh"
    assert (
        installer_script_for("nvidia-container-toolkit", profile)
        == "install_nvidia_container_toolkit_debian.sh"
    )


def test_manual_install_instructions_for_arch() -> None:
    profile = HostProfile("arch-arch-rolling", "arch", "arch", "rolling", "x86_64", True)

    assert "pacman" in manual_install_instructions(profile)


def test_component_is_ready_requires_ok_for_docker_user_access() -> None:
    checks = [CheckResult("docker permissions", "warn", "user may need docker group membership")]

    assert _component_is_ready("docker-user-access", checks) is False


def test_verify_toolkit_host_includes_smoke_test(monkeypatch) -> None:
    monkeypatch.setattr(
        "miner_cli.toolkit.host_checks",
        lambda: [CheckResult("docker cli", "ok", "ready")],
        raising=False,
    )
    monkeypatch.setattr(
        "miner_cli.doctor.host_checks",
        lambda: [CheckResult("docker cli", "ok", "ready")],
    )
    monkeypatch.setattr(
        "miner_cli.doctor.gpu_container_smoke_test",
        lambda progress=None: CheckResult("gpu container smoke test", "ok", "ready"),
    )

    checks = verify_toolkit_host(include_smoke_test=True)

    assert [check.label for check in checks] == ["docker cli", "gpu container smoke test"]


def test_prepare_runtime_rejects_unsupported_engine() -> None:
    checks = prepare_runtime(engine="sglang")

    assert checks == [CheckResult("runtime engine", "fail", "unsupported engine: sglang")]


def test_prepare_runtime_uses_config_file(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: demo",
                "engine: vllm",
                "model: Qwen/Qwen2.5-7B-Instruct",
                "hf_cache: ./hf-cache",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "miner_cli.runtime_prepare.config_checks",
        lambda config: [CheckResult("engine image", "ok", config.image or "default")],
    )
    monkeypatch.setattr(
        "miner_cli.runtime_prepare._pull_image",
        lambda image, progress=None: CheckResult("runtime image pull", "ok", image),
    )

    checks = prepare_runtime(engine="vllm", config_file=config_path)

    labels = [check.label for check in checks]
    assert "runtime engine" in labels
    assert "runtime cache path" in labels
    assert "runtime image pull" in labels


def test_runtime_cache_path_permission_error_returns_failure() -> None:
    class DeniedCachePath:
        def exists(self) -> bool:
            return False

        @property
        def parent(self):
            return self

        def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
            raise PermissionError(13, "Permission denied", "/data")

        def __str__(self) -> str:
            return "/data/huggingface"

    check = _ensure_cache_path(DeniedCachePath())  # type: ignore[arg-type]

    assert check.label == "runtime cache path"
    assert check.status == "fail"
    assert "Permission denied" in check.detail
    assert "/data" in check.detail


def test_runtime_prepare_cli_reports_unsupported_engine() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["runtime", "prepare", "--engine", "sglang"])

    assert result.exit_code == 1
    assert "Unsupported engine" in result.stdout


def test_runtime_prepare_cli_reports_config_errors_without_traceback(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("engine: vllm\nmodel: demo/model\n", encoding="utf-8")

    result = runner.invoke(app, ["runtime", "prepare", "--engine", "vllm", "-f", str(config_path)])

    assert result.exit_code == 1
    assert "runtime config" in result.stdout
    assert "Traceback" not in result.stdout


def test_runtime_prepare_cli_reports_smoke_failure_without_traceback(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr("miner_cli.runtime_prepare.config_checks", lambda config: [])
    monkeypatch.setattr(
        "miner_cli.runtime_prepare._pull_image",
        lambda image, progress=None: CheckResult("runtime image pull", "ok", image),
    )

    def missing_docker(progress=None):
        raise FileNotFoundError(2, "No such file or directory", "docker")

    monkeypatch.setattr("miner_cli.runtime_prepare.gpu_container_smoke_test", missing_docker)
    monkeypatch.setattr(
        "miner_cli.runtime_prepare.engine_container_smoke_test",
        lambda engine, image, progress=None: CheckResult("engine container smoke test", "ok", image),
    )
    monkeypatch.setattr(
        "miner_cli.runtime_prepare._vllm_smoke_test",
        lambda image, progress=None: CheckResult("runtime smoke test", "ok", image),
    )

    result = runner.invoke(app, ["runtime", "prepare", "--engine", "vllm", "--smoke-test"])

    assert result.exit_code == 1
    assert "docker not found" in result.stdout
    assert "Traceback" not in result.stdout


def test_init_cli_warns_on_floating_vllm_image() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            "demo",
            "--engine",
            "vllm",
            "--model",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
    )

    assert result.exit_code == 0
    assert "floating vLLM image" in result.stdout


def test_init_cli_rejects_unsupported_image_policy() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            "demo",
            "--engine",
            "vllm",
            "--model",
            "Qwen/Qwen2.5-7B-Instruct",
            "--image-policy",
            "preview",
        ],
    )

    assert result.exit_code == 1
    assert "Unsupported image policy" in result.stdout


def test_engine_container_smoke_test_warns_for_unknown_engine() -> None:
    check = engine_container_smoke_test("sglang", "example/image:latest")

    assert check.status == "warn"
    assert "no smoke test" in check.detail


def test_engine_container_smoke_test_for_vllm_uses_vllm_binary(monkeypatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_run_logged_command(command, progress=None):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="usage: vllm [-h]\n", stderr="")

    monkeypatch.setattr("miner_cli.runtime_prepare.run_logged_command", fake_run_logged_command)

    check = engine_container_smoke_test("vllm", "example/image:latest")

    assert check.status == "ok"
    assert seen["command"][-2:] == ["example/image:latest", "--help"]
    assert "--entrypoint" not in seen["command"]


def test_summarize_process_output_prefers_real_error_line() -> None:
    completed = __import__("subprocess").CompletedProcess(
        args=["bash", "installer.sh"],
        returncode=1,
        stdout="Reading package lists...\n",
        stderr="E: Unable to locate package docker-ce\n",
    )

    assert _summarize_process_output(completed) == "E: Unable to locate package docker-ce"


def test_post_install_guidance_reports_session_refresh(monkeypatch) -> None:
    monkeypatch.setattr(
        "miner_cli.toolkit.verify_toolkit_host",
        lambda include_smoke_test=False, progress=None: [
            CheckResult("docker daemon", "ok", "ready"),
            CheckResult("docker permissions", "warn", "user needs docker group"),
        ],
    )

    checks = _post_install_guidance()

    assert checks == [
        CheckResult(
            "session refresh",
            "warn",
            "docker group membership may need a new shell; run `newgrp docker` or open a new shell session",
        )
    ]


def test_remediation_steps_for_driver_and_toolkit_failures() -> None:
    checks = [
        CheckResult("nvidia-smi", "fail", "not found"),
        CheckResult("nvidia container toolkit", "fail", "missing"),
        CheckResult("docker nvidia runtime", "fail", "not configured"),
    ]

    steps = remediation_steps(checks)

    assert any("Install the host NVIDIA driver first" in step for step in steps)
    assert any("Run `miner-cli toolkit install` to install NVIDIA Container Toolkit" in step for step in steps)
    assert any("configure Docker's `nvidia` runtime" in step for step in steps)


def test_remediation_steps_distinguish_missing_driver_from_hidden_gpu() -> None:
    missing_driver = remediation_steps([CheckResult("nvidia-smi", "fail", "not found")])
    hidden_gpu = remediation_steps([CheckResult("gpu inventory", "fail", "no GPUs detected")])

    assert any("Install the host NVIDIA driver first" in step for step in missing_driver)
    assert any("no gpu is visible" in step.lower() for step in hidden_gpu)


def test_remediation_steps_distinguish_driver_not_loaded() -> None:
    steps = remediation_steps(
        [
            CheckResult(
                "nvidia-smi",
                "fail",
                "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
            )
        ]
    )

    assert any("kernel module state" in step for step in steps)


def test_remediation_steps_distinguish_driver_too_old_for_container() -> None:
    steps = remediation_steps(
        [
            CheckResult(
                "gpu container smoke test",
                "fail",
                "nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.4",
            )
        ]
    )

    assert any("older than the CUDA requirement" in step for step in steps)


def test_doctor_cli_prints_next_steps_for_host_failures(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(
        "miner_cli.cli.host_checks",
        lambda: [
            CheckResult("docker cli", "fail", "not found"),
            CheckResult("nvidia-smi", "fail", "not found"),
        ],
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Next steps" in result.stdout
    assert "Run `miner-cli toolkit install` to install Docker prerequisites" in result.stdout
    assert "Install the host NVIDIA driver first" in result.stdout


def test_up_cli_prints_next_steps_for_port_conflict(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("name: demo\nengine: vllm\nmodel: demo/model\n", encoding="utf-8")
    monkeypatch.setattr(
        "miner_cli.cli.load_config",
        lambda path: DeploymentConfig(name="demo", engine="vllm", model="demo/model", image="example/image:latest", port=8000),
    )
    monkeypatch.setattr("miner_cli.cli.is_port_in_use", lambda port: True)

    result = runner.invoke(app, ["up", "-f", str(config_path)])

    assert result.exit_code == 1
    assert "Port 8000 is already in use" in result.stdout
    assert "Next steps" in result.stdout
    assert "Choose a different `port:` in the config" in result.stdout


def test_up_cli_prints_next_steps_for_pull_failure(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("name: demo\nengine: vllm\nmodel: demo/model\n", encoding="utf-8")
    config = DeploymentConfig(name="demo", engine="vllm", model="demo/model", image="example/image:latest", port=8000)
    monkeypatch.setattr("miner_cli.cli.load_config", lambda path: config)
    monkeypatch.setattr("miner_cli.cli.is_port_in_use", lambda port: False)
    monkeypatch.setattr("miner_cli.cli.gpu_container_smoke_test", lambda progress=None: CheckResult("gpu container smoke test", "ok", "ready"))
    monkeypatch.setattr("miner_cli.cli.engine_container_smoke_test", lambda engine, image, progress=None: CheckResult("engine container smoke test", "ok", "ready"))
    monkeypatch.setattr("miner_cli.cli.write_deployment_files", lambda cfg, source_config_path=None: object())

    def fake_run_compose(paths, *args, capture_output=False):
        assert capture_output is True
        return subprocess.CompletedProcess(
            args=["docker", "compose", *args],
            returncode=1,
            stdout="",
            stderr="Error response from daemon: manifest for example/image:latest not found",
        )

    monkeypatch.setattr("miner_cli.cli.run_compose", fake_run_compose)

    result = runner.invoke(app, ["up", "-f", str(config_path)])

    assert result.exit_code == 1
    assert "Image pull failed" in result.stdout
    assert "manifest for example/image:latest" in result.stdout
    assert "not found" in result.stdout
    assert "Next steps" in result.stdout
    assert "Run `miner-cli runtime prepare --engine vllm -f" in result.stdout
