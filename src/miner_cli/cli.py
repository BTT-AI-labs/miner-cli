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
from .doctor import CheckResult, config_checks, gpu_container_smoke_test, host_checks

app = typer.Typer(no_args_is_help=True)
console = Console()
STATUS_COLOR = {"ok": "green", "warn": "yellow", "fail": "red"}


def _print_doctor_results(checks: list[CheckResult]) -> None:
    table = Table(title="Miner CLI Doctor")
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


def _require_deployment(name: str):
    paths = deployment_paths(name)
    if paths.compose_path.exists():
        return paths
    console.print(f"[red]Unknown deployment: {name}[/red]")
    raise typer.Exit(1)


@app.command()
def doctor(
    config_file: Path | None = typer.Option(None, "-f", "--file", exists=True, readable=True),
) -> None:
    """Check whether the host is ready for Docker-based model deployment."""
    checks: list[CheckResult] = host_checks()
    if config_file is not None:
        config = load_config(config_file)
        checks.extend(config_checks(config))

    _print_doctor_results(checks)

    failed = [check.label for check in checks if check.status == "fail"]
    if failed:
        raise typer.Exit(1)


@app.command()
def init(
    name: str = typer.Argument(..., help="Deployment name"),
    engine: str = typer.Option(..., help="Inference engine: sglang or vllm"),
    model: str = typer.Option(..., help="Hugging Face model id"),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallel degree / GPU count"),
    port: int = typer.Option(8000, help="Exposed API port"),
    image: str | None = typer.Option(None, help="Override the default engine image"),
    output: Path = typer.Option(Path("."), help="Directory to write the template config into"),
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
    config_file: Path = typer.Option(..., "-f", "--file", exists=True, readable=True),
) -> None:
    """Render compose artifacts without starting the deployment."""
    config = load_config(config_file)
    paths = write_deployment_files(config, source_config_path=config_file)
    console.print(f"Rendered deployment files into [bold]{paths.root}[/bold]")
    console.print(f"Compose file: {paths.compose_path}")


@app.command()
def up(
    config_file: Path = typer.Option(..., "-f", "--file", exists=True, readable=True),
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
        smoke = gpu_container_smoke_test()
        if smoke.status != "ok":
            console.print(f"[red]{smoke.label} failed:[/red] {smoke.detail}")
            raise typer.Exit(1)

    paths = write_deployment_files(config, source_config_path=config_file)

    if pull:
        result = run_compose(paths, "pull")
        if result.returncode != 0:
            raise typer.Exit(result.returncode)

    result = run_compose(paths, "up", "-d")
    if result.returncode != 0:
        raise typer.Exit(result.returncode)

    if wait:
        console.print("Waiting for service readiness...")
        try:
            wait_for_ready(config)
        except TimeoutError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

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


if __name__ == "__main__":
    app()
