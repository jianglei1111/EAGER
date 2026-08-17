from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from eager.config import canonical_json_bytes


class DataFormatError(ValueError):
    """Raised when JSON or JSONL input violates its file contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataFormatError(f"cannot read JSON {path}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterator[Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataFormatError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    except OSError as exc:
        raise DataFormatError(f"cannot read JSONL {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[Any]:
    return list(iter_jsonl(path))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    _atomic_write(path, payload)


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any, *, pretty: bool = True) -> None:
    if pretty:
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    else:
        encoded = canonical_json_bytes(payload) + b"\n"
    _atomic_write(path, encoded)


def atomic_write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                handle.write(canonical_json_bytes(record))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def append_jsonl(path: Path, record: Any, *, durable: bool = True) -> None:
    """Append one canonical JSON record. The caller owns the single-writer contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json_bytes(record) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())


def file_inventory_entry(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    display = path.resolve()
    if root is not None:
        display = display.relative_to(root.resolve())
    return {
        "path": display.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
