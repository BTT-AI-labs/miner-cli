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


def run_logged_command(
    command: list[str],
    progress: ProgressLogger | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if progress is None:
        return subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
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
