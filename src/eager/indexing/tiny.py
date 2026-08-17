from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from eager import __version__
from eager.config import canonical_json_bytes
from eager.indexing.corpus import CorpusSnapshot, CorpusSummary, load_corpus_snapshot
from eager.indexing.errors import IndexValidationError, IndexingError
from eager.indexing.model_registry import (
    ModelPin,
    artifact_manifest_sha256,
    model_pin_sha256,
)
from eager.runtime.io import atomic_write_json, atomic_write_jsonl, sha256_file


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
StableId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^\S+$")]


_ALGORITHM_SPEC = {
    "format": "eager.tiny-hash-v1",
    "normalization": "NFKC-casefold",
    "tokenizer": "unicode-word-v1",
    "weighting": "signed-feature-hash-l2",
}


class IndexArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    path: Literal["vectors.jsonl"] = "vectors.jsonl"
    bytes: int = Field(ge=1)
    sha256: Sha256
    record_count: int = Field(ge=1)


class IndexSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    index_format: Literal["eager.tiny-hash-v1"] = "eager.tiny-hash-v1"
    algorithm_sha256: Sha256
    package_version: str = Field(min_length=1)
    dimensions: int = Field(ge=8, le=4096)
    parameters_sha256: Sha256
    corpus: CorpusSummary
    encoder: ModelPin
    vectors: IndexArtifact


class TinyVectorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    evidence_id: StableId
    paragraph_id: StableId
    sentence_id: StableId
    vector: tuple[float, ...] = Field(min_length=8, max_length=4096)

    @field_validator("vector")
    @classmethod
    def require_finite_vector(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("vector values must be finite")
        return value

    @model_validator(mode="after")
    def validate_evidence_identity(self) -> "TinyVectorRecord":
        if self.evidence_id != self.sentence_id:
            raise ValueError("tiny index evidence_id must equal sentence_id")
        prefix = f"{self.paragraph_id}_s"
        position = self.sentence_id.removeprefix(prefix)
        if not self.sentence_id.startswith(prefix) or not position.isdigit():
            raise ValueError("sentence_id must encode its paragraph_id and position")
        return self


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _parameters_sha256(dimensions: int) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "algorithm_sha256": ALGORITHM_SHA256,
                "dimensions": dimensions,
                "include_title": True,
            }
        )
    ).hexdigest()


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = _TOKEN_PATTERN.findall(normalized)
    return tokens or [normalized]


def _encode_text(text: str, *, dimensions: int) -> tuple[float, ...]:
    values = [0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % dimensions
        sign = 1 if digest[8] & 1 else -1
        values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values[int.from_bytes(digest[:8], "big") % dimensions] = 1
        norm = 1.0
    return tuple(round(value / norm, 12) for value in values)


def encode_tiny_text(text: str, *, dimensions: int) -> tuple[float, ...]:
    """Encode a query with the exact deterministic algorithm used by the fixture index."""

    if dimensions < 8 or dimensions > 4096:
        raise IndexingError("dimensions must be between 8 and 4096")
    return _encode_text(text, dimensions=dimensions)


def _vector_records(snapshot: CorpusSnapshot, *, dimensions: int) -> tuple[TinyVectorRecord, ...]:
    records = [
        TinyVectorRecord(
            evidence_id=unit.evidence_id,
            paragraph_id=unit.paragraph_id,
            sentence_id=unit.sentence_id,
            vector=_encode_text(f"{unit.title}\n{unit.text}", dimensions=dimensions),
        )
        for unit in snapshot.evidence_units
    ]
    return tuple(sorted(records, key=lambda item: item.evidence_id))


ALGORITHM_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "spec": _ALGORITHM_SPEC,
            "token_pattern": _TOKEN_PATTERN.pattern,
            "token_pattern_flags": _TOKEN_PATTERN.flags,
            "tokens_source": inspect.getsource(_tokens),
            "encoder_source": inspect.getsource(_encode_text),
            "vector_records_source": inspect.getsource(_vector_records),
        }
    )
).hexdigest()
_TINY_MANIFEST_SHA256 = artifact_manifest_sha256(())
_TINY_PIN_FIELDS = {
    "name": "tiny_hash_encoder",
    "role": "test_embedding",
    "model_id": "eager/tiny-hash-encoder",
    "revision": ALGORITHM_SHA256[:40],
    "artifact_manifest_sha256": _TINY_MANIFEST_SHA256,
}
TINY_MODEL_PIN = ModelPin(
    **_TINY_PIN_FIELDS,
    artifacts=(),
    pin_sha256=model_pin_sha256(**_TINY_PIN_FIELDS),
)


def _coerce_snapshot(corpus: Path | CorpusSnapshot) -> CorpusSnapshot:
    return corpus if isinstance(corpus, CorpusSnapshot) else load_corpus_snapshot(Path(corpus))


def build_tiny_index(
    corpus: Path | CorpusSnapshot,
    output_dir: Path,
    *,
    dimensions: int = 64,
) -> IndexSummary:
    if dimensions < 8 or dimensions > 4096:
        raise IndexingError("dimensions must be between 8 and 4096")
    snapshot = _coerce_snapshot(corpus)
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise IndexingError(f"index output is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_records = _vector_records(snapshot, dimensions=dimensions)
    vectors_path = output_dir / "vectors.jsonl"
    atomic_write_jsonl(
        vectors_path,
        (record.model_dump(mode="json") for record in vector_records),
    )
    artifact = IndexArtifact(
        bytes=vectors_path.stat().st_size,
        sha256=sha256_file(vectors_path),
        record_count=len(vector_records),
    )
    summary = IndexSummary(
        algorithm_sha256=ALGORITHM_SHA256,
        package_version=__version__,
        dimensions=dimensions,
        parameters_sha256=_parameters_sha256(dimensions),
        corpus=snapshot.summary,
        encoder=TINY_MODEL_PIN,
        vectors=artifact,
    )
    atomic_write_json(output_dir / "summary.json", summary.model_dump(mode="json"))
    return summary


def _read_vector_records(path: Path) -> tuple[TinyVectorRecord, ...]:
    records: list[TinyVectorRecord] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IndexValidationError(
                        f"index vectors cannot contain blank lines: {path}:{line_number}"
                    )
                try:
                    records.append(TinyVectorRecord.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise IndexValidationError(
                        f"invalid vector record at {path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise IndexValidationError(f"cannot read index vectors {path}: {exc}") from exc
    return tuple(records)


@dataclass(frozen=True, slots=True)
class TinyIndexSnapshot:
    summary: IndexSummary
    vectors: tuple[TinyVectorRecord, ...]

    def __post_init__(self) -> None:
        if len(self.vectors) != self.summary.vectors.record_count:
            raise IndexValidationError("tiny index snapshot vector count mismatch")


def load_tiny_index(
    index_dir: Path,
    *,
    corpus: Path | CorpusSnapshot | None = None,
) -> TinyIndexSnapshot:
    """Load a fully validated fixture index for deterministic CPU retrieval."""

    summary = validate_index(index_dir, corpus=corpus, require_corpus=corpus is not None)
    vectors = _read_vector_records(Path(index_dir) / summary.vectors.path)
    return TinyIndexSnapshot(summary=summary, vectors=vectors)


def validate_index(
    index_dir: Path,
    *,
    corpus: Path | CorpusSnapshot | None = None,
    expected_model_pin: ModelPin | None = None,
    require_corpus: bool = False,
) -> IndexSummary:
    index_dir = Path(index_dir)
    summary_path = index_dir / "summary.json"
    try:
        payload: Any = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        summary = IndexSummary.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise IndexValidationError(f"invalid index summary {summary_path}: {exc}") from exc

    if summary.algorithm_sha256 != ALGORITHM_SHA256:
        raise IndexValidationError("index algorithm revision does not match this runtime")
    if summary.parameters_sha256 != _parameters_sha256(summary.dimensions):
        raise IndexValidationError("index parameter hash mismatch")
    if summary.encoder != TINY_MODEL_PIN:
        raise IndexValidationError("tiny index encoder pin does not match this runtime")
    if expected_model_pin is not None and summary.encoder != expected_model_pin:
        raise IndexValidationError("index encoder pin differs from the requested model pin")

    vectors_path = index_dir / summary.vectors.path
    try:
        actual_bytes = vectors_path.stat().st_size
        actual_sha256 = sha256_file(vectors_path)
    except OSError as exc:
        raise IndexValidationError(f"cannot inspect index vectors {vectors_path}: {exc}") from exc
    if actual_bytes != summary.vectors.bytes:
        raise IndexValidationError("index vector byte count mismatch")
    if actual_sha256 != summary.vectors.sha256:
        raise IndexValidationError("index vector hash mismatch")

    vector_records = _read_vector_records(vectors_path)
    if len(vector_records) != summary.vectors.record_count:
        raise IndexValidationError("index vector record count mismatch")
    if len(vector_records) != summary.corpus.evidence_count:
        raise IndexValidationError("index vector count does not match corpus evidence count")
    evidence_ids = [record.evidence_id for record in vector_records]
    if evidence_ids != sorted(evidence_ids) or len(evidence_ids) != len(set(evidence_ids)):
        raise IndexValidationError("index evidence IDs must be unique and sorted")
    for record in vector_records:
        if len(record.vector) != summary.dimensions:
            raise IndexValidationError(
                f"vector dimension mismatch for evidence {record.evidence_id}"
            )
        norm = math.sqrt(sum(value * value for value in record.vector))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise IndexValidationError(
                f"vector is not normalized for evidence {record.evidence_id}"
            )

    if require_corpus and corpus is None:
        raise IndexValidationError("full index validation requires the canonical corpus")
    if corpus is not None:
        snapshot = _coerce_snapshot(corpus)
        if snapshot.summary != summary.corpus:
            raise IndexValidationError("index corpus provenance does not match the supplied corpus")
        expected_records = _vector_records(snapshot, dimensions=summary.dimensions)
        if vector_records != expected_records:
            raise IndexValidationError("index vectors do not match the supplied corpus")
    return summary


__all__ = [
    "ALGORITHM_SHA256",
    "IndexSummary",
    "IndexValidationError",
    "IndexingError",
    "TINY_MODEL_PIN",
    "TinyIndexSnapshot",
    "TinyVectorRecord",
    "build_tiny_index",
    "encode_tiny_text",
    "load_tiny_index",
    "validate_index",
]
