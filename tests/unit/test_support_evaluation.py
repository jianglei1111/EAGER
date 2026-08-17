from __future__ import annotations

import pytest

from eager.evaluation import evaluate_support
from eager.schemas import EvaluationAnnotations, EvidenceUnit, QuestionHandle


def _evidence(paragraph: int, sentence: int) -> EvidenceUnit:
    paragraph_id = f"p{paragraph}"
    sentence_id = f"{paragraph_id}_s{sentence}"
    return EvidenceUnit(
        evidence_id=sentence_id,
        paragraph_id=paragraph_id,
        sentence_id=sentence_id,
        title=f"Title {paragraph}",
        text=f"Evidence {paragraph}:{sentence}",
    )


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wiki", "musique"])
def test_support_evaluator_is_dataset_neutral_and_gold_is_external(dataset: str) -> None:
    handle = QuestionHandle.create(
        dataset=dataset,
        split="development",
        raw_dataset_id="raw-1",
    )
    annotations = EvaluationAnnotations(
        handle_id=handle.handle_id,
        answer="answer",
        supporting_sentence_ids=("p1_s0", "p2_s0"),
        supporting_paragraph_ids=("p1", "p2"),
    )
    metrics = evaluate_support(
        [_evidence(1, 0), _evidence(1, 1), _evidence(3, 0), _evidence(2, 0)],
        annotations,
        limit=3,
    )

    assert metrics.sentence.any_recall is True
    assert metrics.sentence.full_recall is False
    assert metrics.sentence.precision == pytest.approx(1 / 3)
    assert metrics.sentence.recall == 0.5
    assert metrics.paragraph.precision == pytest.approx(1 / 2)
    assert metrics.paragraph.recall == 0.5


def test_missing_gold_granularity_is_reported_as_not_applicable() -> None:
    handle = QuestionHandle.create(
        dataset="musique",
        split="development",
        raw_dataset_id="raw-2",
    )
    metrics = evaluate_support(
        [_evidence(1, 0)],
        EvaluationAnnotations(
            handle_id=handle.handle_id,
            answer="answer",
            supporting_paragraph_ids=("p1",),
        ),
    )
    assert metrics.sentence.any_recall is None
    assert metrics.sentence.precision is None
    assert metrics.paragraph.full_recall is True
