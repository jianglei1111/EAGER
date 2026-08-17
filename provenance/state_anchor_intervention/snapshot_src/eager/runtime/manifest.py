from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from eager.runtime.io import atomic_write_json, file_inventory_entry
from eager.runtime.logging import redact


class ManifestError(ValueError):
    pass


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "status": "available",
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "status": "unavailable",
            "commit": None,
            "branch": None,
            "dirty": None,
        }


def environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def output_inventory(
    root: Path,
    *,
    paths: Iterable[Path] | None = None,
) -> list[dict[str, Any]]:
    selected = list(paths) if paths is not None else [path for path in root.rglob("*") if path.is_file()]
    return [
        file_inventory_entry(path, root=root)
        for path in sorted(selected, key=lambda value: value.resolve().relative_to(root.resolve()).as_posix())
    ]


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    safe = redact(dict(payload))
    if not isinstance(safe, dict):
        raise ManifestError("manifest payload must be an object")
    atomic_write_json(path, safe)
