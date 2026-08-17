from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from eager.data.base import (
    DatasetAdapter,
    DatasetAdapterError,
    DatasetDependencyError,
    require_sequence,
    require_text,
)
from eager.data.context import parse_titled_context, resolve_supporting_facts
from eager.data.identity import encode_relation
from eager.indexing.corpus import CorpusRecord
from eager.schemas import EvaluationAnnotations, InferencePayload, QuestionHandle


def _decode_json_array(value: object, *, field: str) -> Sequence[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DatasetAdapterError(f"{field} contains invalid JSON: {exc}") from exc
    return require_sequence(value, field=field)


class TwoWikiAdapter(DatasetAdapter):
    dataset: ClassVar[str] = "2wiki"
    default_split: ClassVar[str] = "dev"
    _TYPES = frozenset({"bridge_comparison", "comparison", "compositional", "inference"})

    def read_rows(self, path: Path) -> Iterable[Mapping[str, Any]]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise DatasetDependencyError(
                "reading 2Wiki Parquet requires the eager-rag[core] pandas and pyarrow dependencies"
            ) from exc
        try:
            frame = pd.read_parquet(
                path,
                columns=["_id", "type", "question", "context", "supporting_facts", "evidences", "answer"],
            )
        except Exception as exc:  # pandas normalizes multiple parquet backends into different errors
            raise DatasetAdapterError(f"cannot read 2Wiki Parquet {path}: {exc}") from exc
        return frame.to_dict(orient="records")

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
        contexts = parse_titled_context(
            _decode_json_array(row.get("context"), field="context"),
            dataset=self.dataset,
        )
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
        context = _decode_json_array(row.get("context"), field="context")
        supporting_facts = _decode_json_array(row.get("supporting_facts"), field="supporting_facts")
        evidences = _decode_json_array(row.get("evidences"), field="evidences")
        contexts = parse_titled_context(context, dataset=self.dataset)
        support = resolve_supporting_facts(supporting_facts, contexts=contexts)
        gold_type = require_text(row.get("type"), field="type")
        if gold_type not in self._TYPES:
            raise DatasetAdapterError(f"unsupported 2Wiki type: {gold_type}")
        relations: list[str] = []
        for index, raw_relation in enumerate(evidences):
            relation = require_sequence(raw_relation, field=f"evidences[{index}]")
            try:
                relations.append(encode_relation(relation))
            except ValueError as exc:
                raise DatasetAdapterError(f"invalid evidences[{index}]: {exc}") from exc
        return EvaluationAnnotations(
            handle_id=handle.handle_id,
            answer=require_text(row.get("answer"), field="answer", allow_empty=True),
            supporting_sentence_ids=support.sentence_ids,
            supporting_paragraph_ids=support.paragraph_ids,
            supporting_titles=support.titles,
            gold_type=gold_type,
            hop_count=4 if gold_type == "bridge_comparison" else 2,
            gold_relations=relations,
            answerable=True,
        )


__all__ = ["TwoWikiAdapter"]
