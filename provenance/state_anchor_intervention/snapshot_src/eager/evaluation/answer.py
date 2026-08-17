from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    lowered = text.casefold().translate(_PUNCTUATION)
    without_articles = _ARTICLES.sub(" ", lowered)
    return " ".join(without_articles.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    common = Counter(predicted) & Counter(gold)
    matched = sum(common.values())
    if matched == 0:
        return 0.0
    precision = matched / len(predicted)
    recall = matched / len(gold)
    return 2 * precision * recall / (precision + recall)


def containment_accuracy(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction)
    gold = normalize_answer(reference)
    if not gold:
        return float(not predicted)
    return float(gold in predicted)


class AnswerMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    exact_match: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    containment_accuracy: float = Field(ge=0.0, le=1.0)
    matched_reference: str


def evaluate_answer(
    prediction: str,
    reference: str,
    *,
    aliases: Sequence[str] = (),
) -> AnswerMetrics:
    references = tuple(dict.fromkeys((reference, *aliases)))
    scored = [
        (
            exact_match(prediction, item),
            token_f1(prediction, item),
            containment_accuracy(prediction, item),
            item,
        )
        for item in references
    ]
    exact, f1, containment, matched = max(
        scored,
        key=lambda item: (item[0], item[1], item[2], -references.index(item[3])),
    )
    return AnswerMetrics(
        exact_match=exact,
        f1=f1,
        containment_accuracy=containment,
        matched_reference=matched,
    )


__all__ = [
    "AnswerMetrics",
    "containment_accuracy",
    "evaluate_answer",
    "exact_match",
    "normalize_answer",
    "token_f1",
]
