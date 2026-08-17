from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from eager.schemas import InferencePayload


SignalMode = Literal["retrieval_question", "subquestion_chain"]


class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    mode: SignalMode
    reason: str = Field(min_length=1)
    selected_by: str = Field(min_length=1)
    retrieval_score: int = Field(default=0, ge=0)
    chain_score: int = Field(default=0, ge=0)
    evidence: tuple[str, ...] = ()
    fallback: bool = False


class SignalRouter(Protocol):
    def route(self, payload: InferencePayload) -> RouterDecision: ...


_STRONG_COMPARISON = (
    re.compile(r"\b(both|same|different|common|similar)\b", re.IGNORECASE),
    re.compile(r"\bwhich\b[^?]{0,140}\b(or|between)\b", re.IGNORECASE),
    re.compile(
        r"\b(older|younger|earlier|later|larger|smaller|higher|lower|longer|shorter)\b"
        r"[^?]{0,140}\b(than|or|between)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(was|were|is|are)\b[^?]{0,80}\breleased first\b", re.IGNORECASE),
)
_WEAK_COMPARISON = re.compile(
    r"\b(first|last|more|less|closest|nearest|lowest|highest|smallest|largest|most|least)\b",
    re.IGNORECASE,
)
_CHAIN = (
    re.compile(
        r"\b(father|mother|spouse|wife|husband|son|daughter|parent|child|sibling|director|"
        r"composer|author|writer|publisher|producer|creator|founder|owner)\s+of\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(person|actor|actress|artist|writer|director|composer|company|team|film|song|"
        r"album|series|place|city)\s+(who|that|which|whose)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhose\b", re.IGNORECASE),
    re.compile(r"\b(where|when|who|that|which)\b[^?]{0,80}\b(where|when|who|that|which)\b", re.IGNORECASE),
)
_RELATION_TERMS = re.compile(
    r"\b(father|mother|spouse|director|author|writer|creator|founder|birth|death|located|"
    r"member|owner|producer)\b",
    re.IGNORECASE,
)


def score_rule_cues(question: str) -> tuple[int, int, tuple[str, ...]]:
    retrieval_score = 0
    chain_score = 0
    evidence: list[str] = []
    strong = sum(bool(pattern.search(question)) for pattern in _STRONG_COMPARISON)
    if strong:
        retrieval_score += 4 + strong
        evidence.append(f"strong_comparison_or_parallel={strong}")
    weak = len(_WEAK_COMPARISON.findall(question))
    if weak:
        retrieval_score += weak
        evidence.append(f"weak_comparison_terms={weak}")
    chain_hits = sum(bool(pattern.search(question)) for pattern in _CHAIN)
    if chain_hits:
        chain_score += 2 * chain_hits
        evidence.append(f"chain_patterns={chain_hits}")
    lowered = question.casefold()
    of_count = len(re.findall(r"\bof\b", lowered))
    if of_count >= 2 and _RELATION_TERMS.search(question):
        chain_score += 1
        evidence.append(f"relation_with_multiple_of={of_count}")
    if len(question.split()) <= 12 and chain_score == 0:
        retrieval_score += 1
        evidence.append("short_without_explicit_chain")
    if re.search(r"\bor\b", lowered) and re.search(
        r"\b(which|whether|both|same|different|first|earlier|later|older|younger)\b",
        lowered,
    ):
        retrieval_score += 1
        evidence.append("or_with_comparison_context")
    return retrieval_score, chain_score, tuple(evidence)


class RuleRouter:
    def route(self, payload: InferencePayload) -> RouterDecision:
        retrieval_score, chain_score, evidence = score_rule_cues(payload.question)
        if retrieval_score >= 4 and retrieval_score >= chain_score:
            mode: SignalMode = "retrieval_question"
            reason = "comparison or parallel-evidence cues dominate"
        elif chain_score >= 3 or chain_score > retrieval_score:
            mode = "subquestion_chain"
            reason = "ordered slot-filling dependency cues dominate"
        else:
            mode = "retrieval_question"
            reason = "conservative fallback for weakly marked questions"
        return RouterDecision(
            mode=mode,
            reason=reason,
            selected_by="rule_router_v1",
            retrieval_score=retrieval_score,
            chain_score=chain_score,
            evidence=evidence,
        )


def parse_router_output(raw_output: str) -> tuple[SignalMode, str, bool]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = {}
        raw_mode = str(payload.get("signal_type") or payload.get("mode") or "")
        normalized = raw_mode.casefold().replace("-", "_").strip()
        if normalized in {"subquestion", "chain", "subquestion_chain"}:
            return "subquestion_chain", str(payload.get("reason") or "LLM router"), False
        if normalized in {"retrieval", "retrieval_question", "stepwise"}:
            return "retrieval_question", str(payload.get("reason") or "LLM router"), False
    lowered = text.casefold()
    if "subquestion" in lowered or "chain" in lowered:
        return "subquestion_chain", "recovered mode from non-JSON classifier output", True
    return "retrieval_question", "invalid classifier output; conservative fallback", True


class LLMRouter:
    def __init__(self, classifier: Callable[[InferencePayload], str]) -> None:
        self.classifier = classifier

    def route(self, payload: InferencePayload) -> RouterDecision:
        mode, reason, fallback = parse_router_output(self.classifier(payload))
        return RouterDecision(
            mode=mode,
            reason=reason,
            selected_by="llm_router",
            fallback=fallback,
        )


class HybridRouter:
    def __init__(self, llm_router: LLMRouter, *, rule_router: RuleRouter | None = None) -> None:
        self.rule_router = rule_router or RuleRouter()
        self.llm_router = llm_router

    def route(self, payload: InferencePayload) -> RouterDecision:
        rule = self.rule_router.route(payload)
        maximum = max(rule.retrieval_score, rule.chain_score)
        gray_zone = (
            (maximum == 1 and rule.chain_score > 0)
            or abs(rule.retrieval_score - rule.chain_score) <= 1
            or (maximum <= 3 and abs(rule.retrieval_score - rule.chain_score) <= 2)
        )
        if not gray_zone:
            return rule
        llm = self.llm_router.route(payload)
        return llm.model_copy(
            update={
                "selected_by": "hybrid_router",
                "retrieval_score": rule.retrieval_score,
                "chain_score": rule.chain_score,
                "evidence": rule.evidence,
            }
        )


__all__ = [
    "HybridRouter",
    "LLMRouter",
    "RouterDecision",
    "RuleRouter",
    "SignalMode",
    "SignalRouter",
    "parse_router_output",
    "score_rule_cues",
]
