from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from eager.config import canonical_json_bytes
from eager.runtime.io import append_jsonl, iter_jsonl
from eager.runtime.logging import redact


class CheckpointError(RuntimeError):
    pass


class CheckpointMismatchError(CheckpointError):
    pass


@dataclass(frozen=True)
class ProvenanceBinding:
    resolved_config_sha256: str
    data_sha256: str
    index_sha256: str
    model_revision: str
    prompt_sha256: str

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    sequence: int
    stage: str
    binding_sha256: str
    state_sha256: str
    state: Mapping[str, Any]


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._records: list[CheckpointRecord] | None = None

    def _load_records(self) -> list[CheckpointRecord]:
        if self._records is not None:
            return self._records
        if not self.path.exists():
            self._records = []
            return self._records
        output: list[CheckpointRecord] = []
        for raw in iter_jsonl(self.path):
            if not isinstance(raw, dict):
                raise CheckpointError("checkpoint JSONL rows must be objects")
            output.append(CheckpointRecord(**raw))
        self._records = output
        return self._records

    def records(self) -> list[CheckpointRecord]:
        with self._lock:
            return list(self._load_records())

    def append(
        self,
        *,
        checkpoint_id: str,
        stage: str,
        binding: ProvenanceBinding,
        state: Mapping[str, Any],
    ) -> CheckpointRecord:
        if not checkpoint_id or not stage:
            raise CheckpointError("checkpoint_id and stage are required")
        safe_state = redact(dict(state))
        state_sha = hashlib.sha256(canonical_json_bytes(safe_state)).hexdigest()
        binding_sha = binding.sha256()
        with self._lock:
            existing = self._load_records()
            matches = [row for row in existing if row.checkpoint_id == checkpoint_id]
            if matches:
                latest = matches[-1]
                if latest.binding_sha256 != binding_sha or latest.state_sha256 != state_sha:
                    raise CheckpointMismatchError(
                        f"checkpoint {checkpoint_id} already exists with different state"
                    )
                return latest
            record = CheckpointRecord(
                checkpoint_id=checkpoint_id,
                sequence=len(existing),
                stage=stage,
                binding_sha256=binding_sha,
                state_sha256=state_sha,
                state=safe_state,
            )
            append_jsonl(self.path, asdict(record))
            existing.append(record)
            return record

    def resume(self, checkpoint_id: str, binding: ProvenanceBinding) -> CheckpointRecord | None:
        binding_sha = binding.sha256()
        with self._lock:
            matches = [
                row for row in self._load_records() if row.checkpoint_id == checkpoint_id
            ]
            if not matches:
                return None
            latest = matches[-1]
            if latest.binding_sha256 != binding_sha:
                raise CheckpointMismatchError(f"checkpoint {checkpoint_id} provenance does not match")
            return latest
