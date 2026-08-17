from __future__ import annotations

import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from eager.runtime.io import DataFormatError
from eager.schemas import EvaluationAnnotations, InferencePayload, QuestionHandle

if TYPE_CHECKING:
    from eager.indexing.corpus import CorpusRecord, CorpusSnapshot


class DatasetAdapterError(ValueError):
    """Raised when a dataset row or selection violates the adapter contract."""


class DatasetDependencyError(DatasetAdapterError):
    """Raised when an optional dataset reader dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class InferenceBatch:
    """Orchestration handles and engine-safe payloads stored without annotations."""

    handles: tuple[QuestionHandle, ...]
    payloads: tuple[InferencePayload, ...]

    def __post_init__(self) -> None:
        if len(self.handles) != len(self.payloads):
            raise DatasetAdapterError("inference handles and payloads must have equal length")
        handle_ids = [handle.handle_id for handle in self.handles]
        if len(handle_ids) != len(set(handle_ids)):
            raise DatasetAdapterError("inference batch contains duplicate handles")

    def items(self) -> Iterator[tuple[QuestionHandle, InferencePayload]]:
        return iter(zip(self.handles, self.payloads, strict=True))

    def engine_payloads(self) -> tuple[InferencePayload, ...]:
        return self.payloads


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    """Evaluator-only annotation store keyed by opaque handle IDs."""

    by_handle_id: Mapping[str, EvaluationAnnotations]

    def __post_init__(self) -> None:
        copied = dict(self.by_handle_id)
        for key, annotation in copied.items():
            if key != annotation.handle_id:
                raise DatasetAdapterError("evaluation mapping key does not match annotation handle_id")
        object.__setattr__(self, "by_handle_id", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class AdapterBatch:
    """Physically separated inference and evaluation partitions from one source."""

    inference: InferenceBatch
    evaluation: EvaluationBatch

    def __post_init__(self) -> None:
        inference_ids = {handle.handle_id for handle in self.inference.handles}
        evaluation_ids = set(self.evaluation.by_handle_id)
        if inference_ids != evaluation_ids:
            raise DatasetAdapterError("inference and evaluation partitions contain different handles")


def require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetAdapterError(f"{field} must be an object")
    return value


def require_sequence(value: object, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise DatasetAdapterError(f"{field} must be an ordered array")
    return value


def require_text(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
    coerce: bool = False,
) -> str:
    if value is None:
        raise DatasetAdapterError(f"{field} is required")
    if not isinstance(value, str):
        if not coerce:
            raise DatasetAdapterError(f"{field} must be text")
        value = str(value)
    text = value.strip()
    if not allow_empty and not text:
        raise DatasetAdapterError(f"{field} must be non-empty")
    return text


def require_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise DatasetAdapterError(f"{field} must be an integer")
    try:
        return operator.index(value)
    except TypeError:
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value)
        raise DatasetAdapterError(f"{field} must be an integer") from None


class DatasetAdapter(ABC):
    dataset: ClassVar[str]
    default_split: ClassVar[str]

    def __init__(self, *, split: str | None = None) -> None:
        self.split = (split or self.default_split).strip().lower()
        if not self.split:
            raise DatasetAdapterError("split must be non-empty")

    @abstractmethod
    def read_rows(self, path: Path) -> Iterable[Mapping[str, Any]]:
        """Read source rows without adapting their fields."""

    @abstractmethod
    def raw_dataset_id(self, row: Mapping[str, Any]) -> str:
        """Return the exact upstream dataset ID kept only by QuestionHandle."""

    @abstractmethod
    def canonical_question_id(self, row: Mapping[str, Any]) -> str:
        """Return the ID representation used by the tracked split registry."""

    @abstractmethod
    def inference_payload(self, row: Mapping[str, Any]) -> InferencePayload:
        """Map a row to the strict engine-visible allowlist."""

    @abstractmethod
    def evaluation_annotations(
        self,
        row: Mapping[str, Any],
        *,
        handle: QuestionHandle,
    ) -> EvaluationAnnotations:
        """Map a row to evaluator-only labels."""

    @abstractmethod
    def corpus_records(self, row: Mapping[str, Any]) -> tuple["CorpusRecord", ...]:
        """Map public context to canonical corpus records without reading annotations."""

    def adapt_row(
        self,
        row: Mapping[str, Any],
    ) -> tuple[QuestionHandle, InferencePayload, EvaluationAnnotations]:
        handle = QuestionHandle.create(
            dataset=self.dataset,
            split=self.split,
            raw_dataset_id=self.raw_dataset_id(row),
        )
        payload = self.inference_payload(row)
        annotation = self.evaluation_annotations(row, handle=handle)
        if payload.dataset != handle.dataset:
            raise DatasetAdapterError("payload dataset does not match its question handle")
        if annotation.handle_id != handle.handle_id:
            raise DatasetAdapterError("annotation handle does not match its question handle")
        return handle, payload, annotation

    def _select_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        question_ids: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        requested: tuple[str, ...] | None = None
        requested_set: set[str] | None = None
        if question_ids is not None:
            requested = tuple(require_text(value, field="question_ids[]") for value in question_ids)
            if len(requested) != len(set(requested)):
                raise DatasetAdapterError("question_ids contains duplicates")
            requested_set = set(requested)

        selected: dict[str, Mapping[str, Any]] = {}
        source_order: list[str] = []
        seen_source_ids: set[str] = set()
        for raw_row in rows:
            row = require_mapping(raw_row, field="dataset row")
            canonical_id = self.canonical_question_id(row)
            if canonical_id in seen_source_ids:
                raise DatasetAdapterError(f"duplicate source question ID: {canonical_id}")
            seen_source_ids.add(canonical_id)
            if requested_set is not None and canonical_id not in requested_set:
                continue
            selected[canonical_id] = row
            source_order.append(canonical_id)

        order = requested if requested is not None else tuple(source_order)
        missing = [question_id for question_id in order if question_id not in selected]
        if missing:
            preview = ", ".join(missing[:5])
            raise DatasetAdapterError(f"source is missing {len(missing)} requested question IDs: {preview}")
        return tuple(selected[question_id] for question_id in order)

    def adapt_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        question_ids: Sequence[str] | None = None,
    ) -> AdapterBatch:
        selected = self._select_rows(rows, question_ids=question_ids)
        triples = [self.adapt_row(row) for row in selected]

        handles = tuple(item[0] for item in triples)
        payloads = tuple(item[1] for item in triples)
        annotations = {item[0].handle_id: item[2] for item in triples}
        return AdapterBatch(
            inference=InferenceBatch(handles=handles, payloads=payloads),
            evaluation=EvaluationBatch(by_handle_id=annotations),
        )

    def build_corpus_snapshot(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        question_ids: Sequence[str] | None = None,
    ) -> "CorpusSnapshot":
        from eager.indexing.corpus import CorpusRecord, CorpusSnapshot

        selected = self._select_rows(rows, question_ids=question_ids)
        records: list[CorpusRecord] = []
        by_paragraph_id: dict[str, CorpusRecord] = {}
        for row in selected:
            for record in self.corpus_records(row):
                if not isinstance(record, CorpusRecord):
                    raise DatasetAdapterError("corpus_records must return CorpusRecord instances")
                previous = by_paragraph_id.get(record.paragraph_id)
                if previous is None:
                    by_paragraph_id[record.paragraph_id] = record
                    records.append(record)
                elif previous != record:
                    raise DatasetAdapterError(
                        f"conflicting corpus records for paragraph_id: {record.paragraph_id}"
                    )
        return CorpusSnapshot.from_records(records)

    def load(
        self,
        path: Path,
        *,
        question_ids: Sequence[str] | None = None,
    ) -> AdapterBatch:
        try:
            return self.adapt_rows(self.read_rows(path), question_ids=question_ids)
        except DatasetAdapterError:
            raise
        except DataFormatError as exc:
            raise DatasetAdapterError(f"cannot read {self.dataset} source {path}: {exc}") from exc


__all__ = [
    "AdapterBatch",
    "DatasetAdapter",
    "DatasetAdapterError",
    "DatasetDependencyError",
    "EvaluationBatch",
    "InferenceBatch",
    "require_integer",
    "require_mapping",
    "require_sequence",
    "require_text",
]
