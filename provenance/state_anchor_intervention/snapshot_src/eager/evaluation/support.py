from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from eager.schemas import EvaluationAnnotations, EvidenceUnit


class UnitSupportMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    gold_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    any_recall: bool | None
    full_recall: bool | None
    precision: float | None = Field(ge=0.0, le=1.0)
    recall: float | None = Field(ge=0.0, le=1.0)
    f1: float | None = Field(ge=0.0, le=1.0)


class SupportMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    evidence_limit: int | None = Field(default=None, ge=1)
    sentence: UnitSupportMetrics
    paragraph: UnitSupportMetrics


def _unit_metrics(predicted: set[str], gold: set[str]) -> UnitSupportMetrics:
    matched = predicted & gold
    if not gold:
        return UnitSupportMetrics(
            gold_count=0,
            predicted_count=len(predicted),
            matched_count=0,
            any_recall=None,
            full_recall=None,
            precision=None,
            recall=None,
            f1=None,
        )
    precision = len(matched) / len(predicted) if predicted else 0.0
    recall = len(matched) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return UnitSupportMetrics(
        gold_count=len(gold),
        predicted_count=len(predicted),
        matched_count=len(matched),
        any_recall=bool(matched),
        full_recall=gold <= predicted,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def evaluate_support(
    evidence: tuple[EvidenceUnit, ...] | list[EvidenceUnit],
    annotations: EvaluationAnnotations,
    *,
    limit: int | None = None,
) -> SupportMetrics:
    """Join gold only in the evaluator and score both support granularities."""

    selected = tuple(evidence)[:limit]
    predicted_sentences = {item.sentence_id for item in selected}
    predicted_paragraphs = {item.paragraph_id for item in selected}
    return SupportMetrics(
        evidence_limit=limit,
        sentence=_unit_metrics(
            predicted_sentences,
            set(annotations.supporting_sentence_ids),
        ),
        paragraph=_unit_metrics(
            predicted_paragraphs,
            set(annotations.supporting_paragraph_ids),
        ),
    )


__all__ = ["SupportMetrics", "UnitSupportMetrics", "evaluate_support"]
