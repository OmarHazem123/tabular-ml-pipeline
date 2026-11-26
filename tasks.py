#!/usr/bin/env python
"""Cross-platform task runner (stdlib only; no ``make`` required).

Usage::

    uv run python tasks.py <task> [<task> ...] [-- extra args for the last task]

Examples::

    uv run python tasks.py check
    uv run python tasks.py test -- -k leakage -x

The Makefile delegates every target here, so the two can never drift apart.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UV_RUN = ("uv", "run", "--locked")

Command = tuple[str, ...]


@dataclass(frozen=True)
class Task:
    help: str
    commands: tuple[Command, ...] = ()
    action: Callable[[Sequence[str]], int] | None = None
    depends: tuple[str, ...] = field(default=())


def _run(command: Sequence[str]) -> int:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.call(list(command), cwd=ROOT)


def _git_text_files() -> list[Path]:
    """Tracked + untracked-but-not-ignored files, excluding binaries."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    files: list[Path] = []
    for raw_name in filter(None, listing.split(b"\0")):
        path = ROOT / raw_name.decode("utf-8")
        if path.is_file() and b"\0" not in path.read_bytes()[:8192]:
            files.append(path)
    return files


def check_line_endings(_: Sequence[str]) -> int:
    offenders = [path for path in _git_text_files() if b"\r" in path.read_bytes()]
    for path in offenders:
        print(f"CR line ending found: {path.relative_to(ROOT).as_posix()}", file=sys.stderr)
    if offenders:
        return 1
    print("All text files use LF line endings.")
    return 0


def clean(_: Sequence[str]) -> int:
    """Remove tool caches. Never touches data/, artifacts/ or results/."""
    targets = [".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "build", "dist"]
    for name in targets:
        shutil.rmtree(ROOT / name, ignore_errors=True)
    for name in (".coverage", "coverage.xml"):
        (ROOT / name).unlink(missing_ok=True)
    for cache in ROOT.rglob("__pycache__"):
        if ".venv" not in cache.parts:
            shutil.rmtree(cache, ignore_errors=True)
    print("Removed tool caches.")
    return 0


TASKS: dict[str, Task] = {
    "sync": Task("Create/refresh .venv exactly from uv.lock.", (("uv", "sync", "--locked"),)),
    "format": Task(
        "Auto-format and apply safe lint fixes.",
        ((*UV_RUN, "ruff", "format", "."), (*UV_RUN, "ruff", "check", "--fix", ".")),
    ),
    "lint": Task(
        "Ruff lint + format check.",
        ((*UV_RUN, "ruff", "check", "."), (*UV_RUN, "ruff", "format", "--check", ".")),
    ),
    "typecheck": Task("mypy --strict over src, tests and tasks.py.", ((*UV_RUN, "mypy"),)),
    "eol": Task("Fail if any text file contains CR line endings.", action=check_line_endings),
    "test": Task("Offline test suite.", ((*UV_RUN, "pytest"),)),
    "test-network": Task(
        "Tests that download the pinned dataset.", ((*UV_RUN, "pytest", "-m", "network"),)
    ),
    "coverage": Task(
        "Offline tests with coverage report.",
        ((*UV_RUN, "pytest", "--cov=ml_pipeline", "--cov-report=term-missing"),),
    ),
    "check": Task("Full quality gate.", depends=("lint", "typecheck", "eol", "test")),
    "validate-data": Task(
        "Download (if needed) and validate the pinned dataset.",
        ((*UV_RUN, "ml-pipeline", "validate-data"),),
    ),
    "clean": Task("Remove tool caches.", action=clean),
}


def run_task(name: str, extra: Sequence[str]) -> int:
    task = TASKS[name]
    for dependency in task.depends:
        code = run_task(dependency, ())
        if code:
            return code
    if task.action is not None:
        return task.action(extra)
    for index, command in enumerate(task.commands):
        is_last = index == len(task.commands) - 1
        code = _run([*command, *(extra if is_last else ())])
        if code:
            return code
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in args:
        split_at = args.index("--")
        args, extra = args[:split_at], args[split_at + 1 :]

    width = max(map(len, TASKS))
    epilog = "\n".join(f"  {name:<{width}}  {task.help}" for name, task in TASKS.items())
    parser = argparse.ArgumentParser(
        prog="tasks.py",
        description="Project task runner.",
        epilog=f"tasks:\n{epilog}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tasks", nargs="+", choices=sorted(TASKS), metavar="task")
    selected: list[str] = parser.parse_args(args).tasks

    for position, name in enumerate(selected):
        is_last = position == len(selected) - 1
        code = run_task(name, extra if is_last else ())
        if code:
            print(f"Task '{name}' failed with exit code {code}.", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
