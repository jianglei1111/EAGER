from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PathContractError(ValueError):
    """Raised when a configured path escapes its declared ownership boundary."""


def ensure_within(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise PathContractError(f"path escapes base directory: {resolved}") from exc
    return resolved


def resolve_path(
    raw_path: str | Path,
    *,
    base: Path,
    allow_absolute: bool = False,
    allow_outside_base: bool = False,
    must_exist: bool = False,
) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        if not allow_absolute:
            raise PathContractError(f"absolute path is not allowed: {raw}")
        resolved = raw.resolve()
    else:
        resolved = (base / raw).resolve()
    if not allow_outside_base:
        ensure_within(resolved, base)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise PathContractError(f"path is outside repository: {path}") from exc


@dataclass(frozen=True)
class RepositoryLayout:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "RepositoryLayout":
        current = (start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
                return cls(root=candidate)
        raise PathContractError(f"cannot discover repository root from {current}")

    def resolve(self, raw_path: str | Path, *, must_exist: bool = False) -> Path:
        return resolve_path(raw_path, base=self.root, must_exist=must_exist)
