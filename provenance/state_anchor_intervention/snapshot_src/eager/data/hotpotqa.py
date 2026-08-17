from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from eager.data.base import DatasetAdapter, DatasetAdapterError, require_sequence, require_text
from eager.data.context import parse_titled_context, resolve_supporting_facts
from eager.indexing.corpus import CorpusRecord
from eager.runtime.io import read_json
from eager.schemas import EvaluationAnnotations, InferencePayload, QuestionHandle


class HotpotQAAdapter(DatasetAdapter):
    dataset: ClassVar[str] = "hotpotqa"
    default_split: ClassVar[str] = "dev_distractor"

    def read_rows(self, path: Path) -> Iterable[Mapping[str, Any]]:
        payload = read_json(path)
        return require_sequence(payload, field="HotpotQA source")

    def raw_dataset_id(self, row: Mapping[str, Any]) -> str:
        return require_text(row.get("_id"), field="_id")

    def canonical_question_id(self, row: Mapping[str, Any]) -> str:
        return self.raw_dataset_id(row)

    def inference_payload(self, row: Mapping[str, Any]) -> InferencePayload:
        return InferencePayload(
            dataset=self.dataset,
            question=require_text(row.get("question"), field="question"),
        )

    def corpus_records(self, row: Mapping[str, Any]) -> tuple[CorpusRecord, ...]:
        contexts = parse_titled_context(row.get("context"), dataset=self.dataset)
        return tuple(
            CorpusRecord(
                paragraph_id=context.paragraph_id,
                title=context.title,
                text=" ".join(context.sentences),
                sentences=context.sentences,
                source_dataset=self.dataset,
                source_split=self.split,
            )
            for context in contexts
        )

    def evaluation_annotations(
        self,
        row: Mapping[str, Any],
        *,
        handle: QuestionHandle,
    ) -> EvaluationAnnotations:
        contexts = parse_titled_context(row.get("context"), dataset=self.dataset)
        support = resolve_supporting_facts(row.get("supporting_facts"), contexts=contexts)
        gold_type = require_text(row.get("type"), field="type")
        if gold_type not in {"bridge", "comparison"}:
            raise DatasetAdapterError(f"unsupported HotpotQA type: {gold_type}")
        return EvaluationAnnotations(
            handle_id=handle.handle_id,
            answer=require_text(row.get("answer"), field="answer", allow_empty=True),
            supporting_sentence_ids=support.sentence_ids,
            supporting_paragraph_ids=support.paragraph_ids,
            supporting_titles=support.titles,
            gold_type=gold_type,
            hop_count=2,
            answerable=True,
        )


__all__ = ["HotpotQAAdapter"]
