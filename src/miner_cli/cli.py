from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import DEFAULT_DEPLOYMENTS_DIR, load_config, write_template_config
from .deploy import (
    deployment_paths,
    is_port_in_use,
    run_compose,
    wait_for_ready,
    write_deployment_files,
)
from .doctor import config_checks, gpu_container_smoke_test, host_checks
from .preparation import CheckResult, print_results
from .runtime_prepare import SUPPORTED_RUNTIME_ENGINES, prepare_runtime
from .toolkit import install_toolkit, verify_toolkit_host

app = typer.Typer(no_args_is_help=True)
toolkit_app = typer.Typer(no_args_is_help=True)
runtime_app = typer.Typer(no_args_is_help=True)
console = Console()


def _print_results(title: str, checks: list[CheckResult]) -> None:
    print_results(console, title, checks)


def _progress_log(message: str) -> None:
    console.print(f"[cyan]{message}[/cyan]")


def _require_deployment(name: str):
    paths = deployment_paths(name)
    if paths.compose_path.exists():
        return paths
    console.print(f"[red]Unknown deployment: {name}[/red]")
    raise typer.Exit(1)


@app.command()
def doctor(
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "-f", "--file", exists=True, readable=True
    ),
) -> None:
    """Check whether the host is ready for Docker-based model deployment."""
    checks: list[CheckResult] = host_checks()
    if config_file is not None:
        config = load_config(config_file)
        checks.extend(config_checks(config))

    _print_results("Miner CLI Doctor", checks)

    failed = [check.label for check in checks if check.status == "fail"]
    if failed:
        raise typer.Exit(1)


@toolkit_app.command("install")
def toolkit_install() -> None:
    """Install supported host prerequisites for containerized GPU inference."""
    checks = install_toolkit(progress=_progress_log)
    _print_results("Miner CLI Toolkit Install", checks)
    if any(check.status == "fail" for check in checks):
        raise typer.Exit(1)


@toolkit_app.command("verify")
def toolkit_verify(
    smoke_test: bool = typer.Option(False, help="Run the GPU container smoke test"),
) -> None:
    """Verify host prerequisites for containerized GPU inference."""
    checks = verify_toolkit_host(include_smoke_test=smoke_test, progress=_progress_log)
    _print_results("Miner CLI Toolkit Verify", checks)
    if any(check.status == "fail" for check in checks):
        raise typer.Exit(1)


@runtime_app.command("prepare")
def runtime_prepare_command(
    engine: str = typer.Option(..., help="Inference engine to prepare"),
    config_file: Path | None = typer.Option(  # noqa: B008
        None, "-f", "--file", exists=True, readable=True
    ),
    pull: bool = typer.Option(True, help="Pull the runtime image during preparation"),
    smoke_test: bool = typer.Option(False, help="Run heavyweight runtime smoke checks"),
    require_hf_token: bool = typer.Option(False, help="Fail if the configured HF token is unset"),
) -> None:
    """Prepare an engine-specific runtime environment before deployment."""
    checks = prepare_runtime(
        engine=engine,
        config_file=config_file,
        pull=pull,
        smoke_test=smoke_test,
        require_hf_token=require_hf_token,
        progress=_progress_log,
    )
    _print_results("Miner CLI Runtime Prepare", checks)
    if any(check.status == "fail" for check in checks):
        if engine not in SUPPORTED_RUNTIME_ENGINES:
            console.print(
                f"[red]Unsupported engine: {engine}. Supported: {', '.join(sorted(SUPPORTED_RUNTIME_ENGINES))}[/red]"
            )
        raise typer.Exit(1)


@app.command()
def init(
    name: str = typer.Argument(..., help="Deployment name"),
    engine: str = typer.Option(..., help="Inference engine: sglang or vllm"),
    model: str = typer.Option(..., help="Hugging Face model id"),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallel degree / GPU count"),
    port: int = typer.Option(8000, help="Exposed API port"),
    image: str | None = typer.Option(None, help="Override the default engine image"),
    output: Path = typer.Option(  # noqa: B008
        Path("."), help="Directory to write the template config into"
    ),
) -> None:
    """Generate a starter YAML config."""
    path = output / f"{name}.yaml"
    write_template_config(
        path,
        name=name,
        engine=engine,
        model=model,
        tensor_parallel=tensor_parallel,
        port=port,
        image=image,
    )
    console.print(f"Template config written to [bold]{path}[/bold]")


@app.command()
def render(
    config_file: Path = typer.Option(..., "-f", "--file", exists=True, readable=True),  # noqa: B008
) -> None:
    """Render compose artifacts without starting the deployment."""
    config = load_config(config_file)
    paths = write_deployment_files(config, source_config_path=config_file)
    console.print(f"Rendered deployment files into [bold]{paths.root}[/bold]")
    console.print(f"Compose file: {paths.compose_path}")


@app.command()
def up(
    config_file: Path = typer.Option(..., "-f", "--file", exists=True, readable=True),  # noqa: B008
    pull: bool = typer.Option(True, help="Pull the image before starting"),
    wait: bool = typer.Option(True, help="Wait for /v1/models to become healthy"),
    skip_smoke_test: bool = typer.Option(False, help="Skip the GPU container smoke test"),
) -> None:
    """Create or update a deployment and start the container."""
    config = load_config(config_file)
    if is_port_in_use(config.port):
        console.print(f"[red]Port {config.port} is already in use[/red]")
        raise typer.Exit(1)

    if not skip_smoke_test:
        console.print("Running GPU container smoke test...")
        smoke = gpu_container_smoke_test(progress=_progress_log)
        if smoke.status != "ok":
            console.print(f"[red]{smoke.label} failed:[/red] {smoke.detail}")
            console.print(
                "[yellow]Run `miner-cli toolkit verify --smoke-test` or `miner-cli toolkit install` to fix host prerequisites.[/yellow]"
            )
            raise typer.Exit(1)

    paths = write_deployment_files(config, source_config_path=config_file)

    if pull:
        result = run_compose(paths, "pull")
        if result.returncode != 0:
            console.print(
                f"[yellow]Image pull failed. Run `miner-cli runtime prepare --engine {config.engine} -f {config_file}` to verify runtime prerequisites.[/yellow]"
            )
            raise typer.Exit(result.returncode)

    result = run_compose(paths, "up", "-d")
    if result.returncode != 0:
        console.print(
            f"[yellow]Container startup failed. Run `miner-cli runtime prepare --engine {config.engine} -f {config_file}` or `miner-cli toolkit verify` for remediation.[/yellow]"
        )
        raise typer.Exit(result.returncode)

    if wait:
        console.print("Waiting for service readiness...")
        try:
            wait_for_ready(config, progress=_progress_log)
        except TimeoutError as exc:
            console.print(f"[red]{exc}[/red]")
            console.print(
                f"[yellow]If the runtime is not ready, run `miner-cli runtime prepare --engine {config.engine} -f {config_file}` before retrying.[/yellow]"
            )
            raise typer.Exit(1) from exc

    console.print(f"Deployment [bold]{config.name}[/bold] is up")
    console.print(f"Endpoint: http://127.0.0.1:{config.port}/v1")


@app.command()
def status(
    name: str = typer.Argument(..., help="Deployment name"),
) -> None:
    """Show docker compose status for a deployment."""
    paths = _require_deployment(name)
    result = run_compose(paths, "ps")
    raise typer.Exit(result.returncode)


@app.command()
def logs(
    name: str = typer.Argument(..., help="Deployment name"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs"),
) -> None:
    """Show container logs for a deployment."""
    paths = _require_deployment(name)
    args = ["logs"]
    if follow:
        args.append("-f")
    result = run_compose(paths, *args)
    raise typer.Exit(result.returncode)


@app.command()
def stop(name: str = typer.Argument(..., help="Deployment name")) -> None:
    """Stop a deployment."""
    paths = _require_deployment(name)
    result = run_compose(paths, "stop")
    raise typer.Exit(result.returncode)


@app.command()
def restart(name: str = typer.Argument(..., help="Deployment name")) -> None:
    """Restart a deployment."""
    paths = _require_deployment(name)
    result = run_compose(paths, "restart")
    raise typer.Exit(result.returncode)


@app.command()
def rm(
    name: str = typer.Argument(..., help="Deployment name"),
    purge_files: bool = typer.Option(False, help="Also remove rendered deployment files"),
) -> None:
    """Remove a deployment."""
    paths = _require_deployment(name)
    result = run_compose(paths, "down")
    if result.returncode != 0:
        raise typer.Exit(result.returncode)
    if purge_files:
        shutil.rmtree(paths.root, ignore_errors=True)
    console.print(f"Removed deployment [bold]{name}[/bold]")


@app.command(name="list")
def list_deployments() -> None:
    """List known deployment directories."""
    root = DEFAULT_DEPLOYMENTS_DIR
    table = Table(title="Miner CLI Deployments")
    table.add_column("Name")
    table.add_column("Path")
    if not root.exists():
        console.print(table)
        return
    for item in sorted(root.iterdir()):
        if item.is_dir() and (item / "compose.yaml").exists():
            table.add_row(item.name, str(item))
    console.print(table)


app.add_typer(toolkit_app, name="toolkit")
app.add_typer(runtime_app, name="runtime")


if __name__ == "__main__":
    app()
