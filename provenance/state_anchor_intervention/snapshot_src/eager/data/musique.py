from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

from eager.data.base import (
    DatasetAdapter,
    DatasetAdapterError,
    require_integer,
    require_mapping,
    require_sequence,
    require_text,
)
from eager.data.identity import make_paragraph_id
from eager.indexing.corpus import CorpusRecord
from eager.runtime.io import iter_jsonl
from eager.schemas import (
    EvaluationAnnotations,
    EvaluationDecompositionStep,
    InferencePayload,
    QuestionHandle,
)


_PLACEHOLDER_RE = re.compile(r"#(\d+)")
_TYPE_RE = re.compile(r"^(\d+)hop")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_musique_sentences(text: str) -> tuple[str, ...]:
    """Reproduce the deterministic sentence view used by the legacy shared corpus."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ()
    sentences = tuple(
        part.strip()
        for part in _SENTENCE_BOUNDARY_RE.split(normalized)
        if part.strip()
    )
    return sentences or (normalized,)


class MusiqueAdapter(DatasetAdapter):
    dataset: ClassVar[str] = "musique"
    default_split: ClassVar[str] = "ans_dev"

    def read_rows(self, path: Path) -> Iterable[Mapping[str, Any]]:
        return iter_jsonl(path)

    def raw_dataset_id(self, row: Mapping[str, Any]) -> str:
        return require_text(row.get("id"), field="id")

    def canonical_question_id(self, row: Mapping[str, Any]) -> str:
        raw_id = self.raw_dataset_id(row)
        return raw_id if raw_id.startswith("musique_") else f"musique_{raw_id}"

    def inference_payload(self, row: Mapping[str, Any]) -> InferencePayload:
        return InferencePayload(
            dataset=self.dataset,
            question=require_text(row.get("question"), field="question"),
        )

    def corpus_records(self, row: Mapping[str, Any]) -> tuple[CorpusRecord, ...]:
        records: list[CorpusRecord] = []
        for position, raw_paragraph in enumerate(
            require_sequence(row.get("paragraphs"), field="paragraphs")
        ):
            paragraph = require_mapping(raw_paragraph, field=f"paragraphs[{position}]")
            title = require_text(paragraph.get("title"), field=f"paragraphs[{position}].title")
            paragraph_text = require_text(
                paragraph.get("paragraph_text"), field=f"paragraphs[{position}].paragraph_text"
            )
            sentences = split_musique_sentences(paragraph_text)
            records.append(
                CorpusRecord(
                    paragraph_id=make_paragraph_id(
                        dataset=self.dataset,
                        title=title,
                        content=sentences,
                    ),
                    title=title,
                    text=" ".join(sentences),
                    sentences=sentences,
                    source_dataset=self.dataset,
                    source_split=self.split,
                )
            )
        if not records:
            raise DatasetAdapterError("paragraphs must contain at least one paragraph")
        return tuple(records)

    def evaluation_annotations(
        self,
        row: Mapping[str, Any],
        *,
        handle: QuestionHandle,
    ) -> EvaluationAnnotations:
        raw_id = self.raw_dataset_id(row)
        type_name = raw_id.removeprefix("musique_").split("__", 1)[0]
        type_match = _TYPE_RE.match(type_name)
        if type_match is None:
            raise DatasetAdapterError(f"cannot derive MuSiQue hop count from id: {raw_id}")

        paragraph_ids_by_index: dict[int, str] = {}
        supporting_paragraph_ids: list[str] = []
        supporting_titles: list[str] = []
        for position, raw_paragraph in enumerate(
            require_sequence(row.get("paragraphs"), field="paragraphs")
        ):
            paragraph = require_mapping(raw_paragraph, field=f"paragraphs[{position}]")
            source_index = require_integer(paragraph.get("idx"), field=f"paragraphs[{position}].idx")
            if source_index < 0:
                raise DatasetAdapterError(f"paragraphs[{position}].idx must be non-negative")
            if source_index in paragraph_ids_by_index:
                raise DatasetAdapterError(f"duplicate MuSiQue paragraph idx: {source_index}")
            title = require_text(paragraph.get("title"), field=f"paragraphs[{position}].title")
            paragraph_text = require_text(
                paragraph.get("paragraph_text"), field=f"paragraphs[{position}].paragraph_text"
            )
            sentences = split_musique_sentences(paragraph_text)
            paragraph_id = make_paragraph_id(
                dataset=self.dataset,
                title=title,
                content=sentences,
            )
            paragraph_ids_by_index[source_index] = paragraph_id
            is_supporting = paragraph.get("is_supporting")
            if not isinstance(is_supporting, bool):
                raise DatasetAdapterError(f"paragraphs[{position}].is_supporting must be boolean")
            if is_supporting:
                supporting_paragraph_ids.append(paragraph_id)
                supporting_titles.append(title)
        if not supporting_paragraph_ids:
            raise DatasetAdapterError("MuSiQue row has no supporting paragraph")

        decomposition_rows = require_sequence(
            row.get("question_decomposition"), field="question_decomposition"
        )
        source_step_ids: list[str] = []
        for index, raw_step in enumerate(decomposition_rows):
            step = require_mapping(raw_step, field=f"question_decomposition[{index}]")
            source_step_ids.append(
                require_text(
                    step.get("id"),
                    field=f"question_decomposition[{index}].id",
                    coerce=True,
                )
            )

        decomposition: list[EvaluationDecompositionStep] = []
        for index, raw_step in enumerate(decomposition_rows):
            step = require_mapping(raw_step, field=f"question_decomposition[{index}]")
            question = require_text(
                step.get("question"), field=f"question_decomposition[{index}].question"
            )
            dependencies: list[str] = []
            for raw_ordinal in _PLACEHOLDER_RE.findall(question):
                ordinal = int(raw_ordinal)
                if ordinal < 1 or ordinal > index:
                    raise DatasetAdapterError(
                        f"question_decomposition[{index}] references non-prior step #{ordinal}"
                    )
                dependencies.append(source_step_ids[ordinal - 1])
            support_index = require_integer(
                step.get("paragraph_support_idx"),
                field=f"question_decomposition[{index}].paragraph_support_idx",
            )
            support_id = paragraph_ids_by_index.get(support_index)
            if support_id is None:
                raise DatasetAdapterError(
                    f"question_decomposition[{index}] references absent paragraph idx {support_index}"
                )
            relation = question.split(">>", 1)[1].strip() if ">>" in question else None
            decomposition.append(
                EvaluationDecompositionStep(
                    step_id=source_step_ids[index],
                    question=question,
                    answer=require_text(
                        step.get("answer"),
                        field=f"question_decomposition[{index}].answer",
                        allow_empty=True,
                    ),
                    depends_on=dependencies,
                    supporting_paragraph_ids=[support_id],
                    relation=relation or None,
                )
            )

        expected_hops = int(type_match.group(1))
        if len(decomposition) != expected_hops:
            raise DatasetAdapterError(
                f"MuSiQue id declares {expected_hops} hops but has {len(decomposition)} decomposition steps"
            )
        answerable = row.get("answerable")
        if not isinstance(answerable, bool):
            raise DatasetAdapterError("answerable must be boolean")
        return EvaluationAnnotations(
            handle_id=handle.handle_id,
            answer=require_text(row.get("answer"), field="answer", allow_empty=True),
            answer_aliases=[
                require_text(alias, field=f"answer_aliases[{index}]")
                for index, alias in enumerate(
                    require_sequence(row.get("answer_aliases", []), field="answer_aliases")
                )
            ],
            supporting_paragraph_ids=supporting_paragraph_ids,
            supporting_titles=supporting_titles,
            gold_type=type_name,
            hop_count=expected_hops,
            official_decomposition=decomposition,
            answerable=answerable,
        )


__all__ = ["MusiqueAdapter", "split_musique_sentences"]
