from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

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
from eager.content_identity import (
    make_paragraph_id,
    make_sentence_id,
    normalize_content_text,
)
from eager.indexing.errors import CorpusValidationError
from eager.runtime.io import sha256_file
from eager.schemas import EvidenceUnit


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StableId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^\S+$")]
Name = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CorpusId = Annotated[str, StringConstraints(pattern=r"^corpus_[0-9a-f]{32}$")]


class CorpusRecord(BaseModel):
    """The only paragraph fields that may enter an inference index."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    paragraph_id: StableId
    title: NonEmptyText
    text: NonEmptyText
    sentences: tuple[str, ...]
    source_dataset: Name
    source_split: Name

    @field_validator("source_dataset", "source_split")
    @classmethod
    def normalize_source_names(cls, value: str) -> str:
        return value.lower()

    @field_validator("title", "text")
    @classmethod
    def canonicalize_content_text(cls, value: str) -> str:
        return normalize_content_text(value)

    @field_validator("sentences", mode="before")
    @classmethod
    def require_ordered_sentences(cls, value: Any) -> Any:
        if isinstance(value, (str, bytes, bytearray, set, frozenset, dict)):
            raise ValueError("sentences must be an ordered array")
        return value

    @field_validator("sentences")
    @classmethod
    def canonicalize_sentences(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_content_text(sentence) for sentence in value)

    @model_validator(mode="after")
    def validate_content_identity(self) -> "CorpusRecord":
        if self.sentences:
            joined = " ".join(self.sentences)
            if normalize_content_text(joined) != self.text:
                raise ValueError("text must contain exactly the normalized sentence content")
            identity_content: str | tuple[str, ...] = self.sentences
        else:
            identity_content = self.text
        expected = make_paragraph_id(
            dataset=self.source_dataset,
            title=self.title,
            content=identity_content,
        )
        if self.paragraph_id != expected:
            raise ValueError(
                f"paragraph_id does not match content: expected {expected}, got {self.paragraph_id}"
            )
        return self


class CorpusSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    package_version: str = Field(min_length=1)
    contract_sha256: Sha256
    parameters_sha256: Sha256
    corpus_id: CorpusId
    input_bytes: int = Field(ge=0)
    input_sha256: Sha256
    canonical_bytes: int = Field(ge=0)
    canonical_sha256: Sha256
    paragraph_count: int = Field(ge=1)
    evidence_count: int = Field(ge=1)
    sentence_count: int = Field(ge=0)
    empty_sentence_count: int = Field(ge=0)
    sources: tuple["CorpusSource", ...] = Field(min_length=1)


class CorpusSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    dataset: Name
    split: Name


def _canonical_corpus_stats(records: tuple[CorpusRecord, ...]) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    for record in sorted(records, key=lambda item: item.paragraph_id):
        line = canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        digest.update(line)
        byte_count += len(line)
    return byte_count, digest.hexdigest()


def _evidence_units(records: tuple[CorpusRecord, ...]) -> tuple[EvidenceUnit, ...]:
    units: list[EvidenceUnit] = []
    for record in records:
        texts = record.sentences or (record.text,)
        for position, text in enumerate(texts):
            if not text:
                continue
            sentence_id = make_sentence_id(record.paragraph_id, position)
            units.append(
                EvidenceUnit(
                    evidence_id=sentence_id,
                    paragraph_id=record.paragraph_id,
                    title=record.title,
                    sentence_id=sentence_id,
                    text=text,
                )
            )
    return tuple(units)


_CORPUS_PARAMETERS = {
    "canonical_order": "paragraph_id",
    "content_id": "sha256(dataset,title,normalized-segments)[:32]",
    "evidence_id": "paragraph_id + sentence-position",
    "empty_sentence_slots": "preserved-in-identity-skipped-in-evidence",
    "paragraph_fallback": "single-s0-evidence",
    "text_normalization": "NFKC-whitespace",
}
CORPUS_PARAMETERS_SHA256 = hashlib.sha256(
    canonical_json_bytes(_CORPUS_PARAMETERS)
).hexdigest()
CORPUS_CONTRACT_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "parameters": _CORPUS_PARAMETERS,
            "record_schema": CorpusRecord.model_json_schema(),
            "record_source": inspect.getsource(CorpusRecord),
            "identity_module_source": inspect.getsource(
                sys.modules[make_paragraph_id.__module__]
            ),
            "evidence_source": inspect.getsource(_evidence_units),
        }
    )
).hexdigest()


def _make_summary(
    records: tuple[CorpusRecord, ...],
    *,
    input_bytes: int | None = None,
    input_sha256: str | None = None,
) -> CorpusSummary:
    canonical_bytes, canonical_sha256 = _canonical_corpus_stats(records)
    if input_bytes is None:
        input_bytes = canonical_bytes
    if input_sha256 is None:
        input_sha256 = canonical_sha256
    sources = tuple(
        CorpusSource(dataset=dataset, split=split)
        for dataset, split in sorted(
            {(record.source_dataset, record.source_split) for record in records}
        )
    )
    empty_sentence_count = sum(
        1
        for record in records
        for sentence in record.sentences
        if not sentence
    )
    evidence_count = sum(
        sum(bool(sentence) for sentence in record.sentences)
        if record.sentences
        else 1
        for record in records
    )
    return CorpusSummary(
        package_version=__version__,
        contract_sha256=CORPUS_CONTRACT_SHA256,
        parameters_sha256=CORPUS_PARAMETERS_SHA256,
        corpus_id=f"corpus_{canonical_sha256[:32]}",
        input_bytes=input_bytes,
        input_sha256=input_sha256,
        canonical_bytes=canonical_bytes,
        canonical_sha256=canonical_sha256,
        paragraph_count=len(records),
        evidence_count=evidence_count,
        sentence_count=evidence_count,
        empty_sentence_count=empty_sentence_count,
        sources=sources,
    )


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """Validated paragraphs and their deterministic sentence-level evidence view."""

    records: tuple[CorpusRecord, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    summary: CorpusSummary

    def __post_init__(self) -> None:
        if not self.records:
            raise CorpusValidationError("corpus must contain at least one record")
        paragraph_ids = [record.paragraph_id for record in self.records]
        if len(paragraph_ids) != len(set(paragraph_ids)):
            raise CorpusValidationError("corpus contains duplicate paragraph_id values")
        expected_units = _evidence_units(self.records)
        if self.evidence_units != expected_units:
            raise CorpusValidationError("evidence_units do not match corpus records")
        expected_summary = _make_summary(
            self.records,
            input_bytes=self.summary.input_bytes,
            input_sha256=self.summary.input_sha256,
        )
        if self.summary != expected_summary:
            raise CorpusValidationError("corpus summary does not match corpus records")

    @classmethod
    def from_records(
        cls,
        records: tuple[CorpusRecord, ...] | list[CorpusRecord],
        *,
        input_bytes: int | None = None,
        input_sha256: str | None = None,
    ) -> "CorpusSnapshot":
        frozen_records = deduplicate_corpus_records(records)
        if not frozen_records:
            raise CorpusValidationError("corpus must contain at least one record")
        summary = _make_summary(
            frozen_records,
            input_bytes=input_bytes,
            input_sha256=input_sha256,
        )
        return cls(
            records=frozen_records,
            evidence_units=_evidence_units(frozen_records),
            summary=summary,
        )


def deduplicate_corpus_records(
    records: tuple[CorpusRecord, ...] | list[CorpusRecord],
) -> tuple[CorpusRecord, ...]:
    output: list[CorpusRecord] = []
    by_id: dict[str, CorpusRecord] = {}
    for record in records:
        previous = by_id.get(record.paragraph_id)
        if previous is None:
            by_id[record.paragraph_id] = record
            output.append(record)
        elif previous != record:
            raise CorpusValidationError(
                f"conflicting records for paragraph_id: {record.paragraph_id}"
            )
    return tuple(output)


def load_corpus_snapshot(path: Path) -> CorpusSnapshot:
    path = Path(path)
    try:
        input_bytes = path.stat().st_size
        input_sha256 = sha256_file(path)
    except OSError as exc:
        raise CorpusValidationError(f"cannot inspect corpus {path}: {exc}") from exc

    records: list[CorpusRecord] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise CorpusValidationError(
                        f"canonical corpus cannot contain blank lines: {path}:{line_number}"
                    )
                try:
                    payload = json.loads(line)
                    records.append(CorpusRecord.model_validate(payload))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise CorpusValidationError(
                        f"invalid corpus record at {path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise CorpusValidationError(f"cannot read corpus {path}: {exc}") from exc
    seen_ids: set[str] = set()
    for record in records:
        if record.paragraph_id in seen_ids:
            raise CorpusValidationError(f"duplicate paragraph_id: {record.paragraph_id}")
        seen_ids.add(record.paragraph_id)
    return CorpusSnapshot.from_records(
        records,
        input_bytes=input_bytes,
        input_sha256=input_sha256,
    )


def validate_corpus(path: Path) -> CorpusSummary:
    return load_corpus_snapshot(path).summary


__all__ = [
    "CORPUS_CONTRACT_SHA256",
    "CORPUS_PARAMETERS_SHA256",
    "CorpusRecord",
    "CorpusSource",
    "CorpusSnapshot",
    "CorpusSummary",
    "CorpusValidationError",
    "deduplicate_corpus_records",
    "load_corpus_snapshot",
    "validate_corpus",
]
