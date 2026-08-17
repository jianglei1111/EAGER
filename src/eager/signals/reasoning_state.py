from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from eager.config import canonical_json_bytes
from eager.runtime.llm_client import ChatMessage, OpenAICompatibleClient
from eager.schemas import (
    ControllerState,
    EvidenceUnit,
    InferencePayload,
    RetrievalState,
    SignalDecision,
    assert_annotation_isolation,
)
from eager.signals._identity import decision_id
from eager.signals.retrieval_question import RetrievalQuestionController


_GROUNDING_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "which",
        "who",
        "with",
    }
)

_UNSUPPORTED_OPERATION_PATTERN = re.compile(
    r"\b(?:"
    r"(?:born|released|published|founded)\s+(?:earlier|later)|"
    r"(?:older|younger|earlier|later|higher|lower|more|less)\s+than|"
    r"came\s+out\s+(?:before|after)|"
    r"(?:therefore|thus|hence)|"
    r"(?:this|which)\s+(?:means|indicates|shows)"
    r")\b",
    re.IGNORECASE,
)


def _grounding_tokens(text: str) -> frozenset[str]:
    normalized = text.casefold().replace("\u2019", "'")
    return frozenset(
        token
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if token not in _GROUNDING_STOPWORDS and (len(token) > 1 or token.isdigit())
    )


def _is_directly_grounded(
    reasoning_state: str,
    evidence: Sequence[EvidenceUnit],
) -> bool:
    if _UNSUPPORTED_OPERATION_PATTERN.search(reasoning_state):
        return False
    state_tokens = _grounding_tokens(reasoning_state)
    if not state_tokens:
        return False
    for item in evidence:
        source_tokens = _grounding_tokens(f"{item.title} {item.text}")
        covered = state_tokens & source_tokens
        required_overlap = min(3, len(state_tokens))
        if len(covered) >= required_overlap and len(covered) / len(state_tokens) >= 0.65:
            return True
    return False


def _repeats_previous_state(
    reasoning_state: str,
    previous_states: Sequence[str],
) -> bool:
    state_tokens = _grounding_tokens(reasoning_state)
    if not state_tokens:
        return False
    for previous in previous_states:
        previous_tokens = _grounding_tokens(previous)
        union = state_tokens | previous_tokens
        if union and len(state_tokens & previous_tokens) / len(union) >= 0.85:
            return True
    return False


def _answer_slot_terms(question: str) -> frozenset[str]:
    normalized = re.sub(r"\s+", " ", question).strip().rstrip("?.!")
    direct_wh_question = re.match(
        r"^(?:who|what|where|when)\s+(?:is|was|are|were)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    possessive = (
        re.search(
            r"(?:'s|s')\s+([A-Za-z][A-Za-z -]{0,40})$",
            normalized,
            flags=re.IGNORECASE,
        )
        if direct_wh_question is not None
        else None
    )
    if possessive is not None:
        return _grounding_tokens(possessive.group(1))
    relational = re.match(
        r"^(?:who|what|which)\s+(?:is|was|are|were)\s+"
        r"(?:the\s+)?([A-Za-z][A-Za-z -]{0,40}?)\s+of\s+",
        normalized,
        flags=re.IGNORECASE,
    )
    return _grounding_tokens(relational.group(1)) if relational is not None else frozenset()


def _fills_question_answer_slot(question: str, reasoning_state: str) -> bool:
    target_terms = _answer_slot_terms(question)
    return bool(target_terms and target_terms <= _grounding_tokens(reasoning_state))


def compose_reasoning_signal(question: str, reasoning_state: str) -> str:
    question = re.sub(r"\s+", " ", question).strip()
    reasoning_state = re.sub(r"\s+", " ", reasoning_state).strip()
    if not question or not reasoning_state:
        raise ValueError("question and reasoning_state must be non-empty")
    return f"{question}\n{reasoning_state}"


def extract_reasoning_state(
    signal_text: str | None,
    *,
    question: str | None = None,
) -> str:
    if not signal_text or "\n" not in signal_text:
        return ""
    normalized_question = re.sub(r"\s+", " ", question or "").strip()
    prefix = f"{normalized_question}\n" if normalized_question else ""
    if prefix:
        if not signal_text.startswith(prefix):
            return ""
        reasoning_state = signal_text[len(prefix) :]
    else:
        reasoning_state = signal_text.split("\n", 1)[1]
    return re.sub(
        r"\s+",
        " ",
        reasoning_state,
    ).strip()


class ReasoningStateGenerator(Protocol):
    def generate(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        iteration: int,
    ) -> str | "ReasoningStateGeneration": ...


@dataclass(frozen=True, slots=True)
class ReasoningStateGeneration:
    text: str
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    call_records: tuple[Mapping[str, Any], ...] = ()
    fallback: bool = False


class OpenAIReasoningStateGenerator:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        system_prompt: str,
        evidence_limit: int = 10,
        evidence_char_limit: int = 700,
        evidence_window_mode: Literal["oldest", "latest_balanced"] = "oldest",
    ) -> None:
        prompt = system_prompt.strip()
        if not prompt:
            raise ValueError("reasoning-state system prompt must be non-empty")
        self.client = client
        self.system_prompt = prompt
        self.system_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.evidence_limit = evidence_limit
        self.evidence_char_limit = evidence_char_limit
        self.evidence_window_mode = evidence_window_mode

    @staticmethod
    def _unique_evidence(items: Sequence[EvidenceUnit]) -> tuple[EvidenceUnit, ...]:
        selected: list[EvidenceUnit] = []
        seen: set[str] = set()
        for item in items:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            selected.append(item)
        return tuple(selected)

    def _prompt_evidence(self, retrieval_state: RetrievalState) -> tuple[EvidenceUnit, ...]:
        if self.evidence_window_mode == "oldest":
            return retrieval_state.retained_evidence[: self.evidence_limit]

        current = self._unique_evidence(retrieval_state.current_evidence)
        current_ids = {item.evidence_id for item in current}
        history = self._unique_evidence(
            tuple(
                item
                for item in retrieval_state.retained_evidence
                if item.evidence_id not in current_ids
            )
        )
        current_quota = (self.evidence_limit + 1) // 2
        history_quota = self.evidence_limit - current_quota
        selected = list(current[:current_quota])
        selected.extend(history[:history_quota])
        seen = {item.evidence_id for item in selected}
        for pool in (current[current_quota:], history[history_quota:]):
            for item in pool:
                if len(selected) >= self.evidence_limit:
                    break
                if item.evidence_id not in seen:
                    selected.append(item)
                    seen.add(item.evidence_id)
        return tuple(selected)

    def _ordered_prompt_evidence(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
    ) -> tuple[EvidenceUnit, ...]:
        selected = self._prompt_evidence(retrieval_state)
        normalized_question = re.sub(r"\s+", " ", payload.question).casefold()
        return tuple(
            item
            for _, item in sorted(
                enumerate(selected),
                key=lambda pair: (
                    0
                    if re.sub(r"\s+", " ", pair[1].title).casefold()
                    in normalized_question
                    else 1,
                    pair[0],
                ),
            )
        )

    def _user_prompt(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        iteration: int,
    ) -> str:
        evidence = [
            {
                "title": item.title,
                "text": item.text[: self.evidence_char_limit],
            }
            for item in self._ordered_prompt_evidence(payload, retrieval_state)
        ]
        previous_states = self._previous_states(payload, retrieval_state)
        request = {
            "original_question": payload.question,
            "iteration": iteration,
            "previous_reasoning_states": previous_states,
            "retained_evidence": evidence,
        }
        assert_annotation_isolation(request)
        return json.dumps(request, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _previous_states(
        payload: InferencePayload,
        retrieval_state: RetrievalState,
    ) -> tuple[str, ...]:
        return tuple(
            state
            for decision in retrieval_state.signal_history
            if decision.controller_type == "reasoning_state"
            for state in (
                extract_reasoning_state(
                    decision.signal_text,
                    question=payload.question,
                ),
            )
            if state
        )

    @staticmethod
    def _parse_text(raw_output: str) -> str:
        text = raw_output.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ""
        if not isinstance(payload, dict):
            return ""
        if set(payload) != {"reasoning_state"}:
            return ""
        value = payload.get("reasoning_state")
        if not isinstance(value, str):
            return ""
        reasoning_state = re.sub(
            r"\s+",
            " ",
            value.strip(),
        )
        if re.search(r"\b(?:the|final)\s+answer\s+is\b", reasoning_state, re.IGNORECASE):
            return ""
        return reasoning_state

    @staticmethod
    def _is_explicit_empty(raw_output: str) -> bool:
        try:
            payload = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            return False
        return (
            isinstance(payload, dict)
            and set(payload) == {"reasoning_state"}
            and isinstance(payload.get("reasoning_state"), str)
            and not payload["reasoning_state"].strip()
        )

    def generate(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        iteration: int,
    ) -> ReasoningStateGeneration:
        user_prompt = self._user_prompt(payload, retrieval_state, iteration)
        prompt_evidence = self._ordered_prompt_evidence(payload, retrieval_state)
        response = self.client.complete(
            (
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ),
            temperature=0.0,
            max_completion_tokens=300,
        )
        generated = self._parse_text(response.content)
        validation_status = "accepted"
        if not generated:
            validation_status = (
                "empty" if self._is_explicit_empty(response.content) else "parse_or_schema_error"
            )
        if generated and _fills_question_answer_slot(payload.question, generated):
            generated = ""
            validation_status = "fills_answer_slot"
        if generated and not _is_directly_grounded(generated, prompt_evidence):
            generated = ""
            validation_status = "ungrounded_or_unsupported_operation"
        if generated and _repeats_previous_state(
            generated,
            self._previous_states(payload, retrieval_state),
        ):
            generated = ""
            validation_status = "repeated"
        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0
        request_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "system_prompt_sha256": self.system_prompt_sha256,
                    "user_prompt": user_prompt,
                }
            )
        ).hexdigest()
        return ReasoningStateGeneration(
            text=generated,
            llm_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_records=(
                {
                    "kind": "reasoning_state_generation",
                    "model": response.model,
                    "prompt_sha256": request_sha256,
                    "system_prompt_sha256": self.system_prompt_sha256,
                    "response_sha256": response.response_sha256,
                    "request_id": response.request_id,
                    "finish_reason": response.finish_reason,
                    "attempts": response.attempts,
                    "elapsed_seconds": response.elapsed_seconds,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "iteration": iteration,
                    "validation_status": validation_status,
                },
            ),
            fallback=not generated,
        )


@dataclass(frozen=True, slots=True)
class StaticReasoningStateGenerator:
    states: tuple[str, ...]

    def __init__(self, states: Sequence[str]) -> None:
        cleaned = tuple(state.strip() for state in states)
        if not cleaned or any(not state for state in cleaned):
            raise ValueError("static reasoning states must be non-empty")
        object.__setattr__(self, "states", cleaned)

    def generate(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        iteration: int,
    ) -> str:
        index = min(max(0, iteration - 2), len(self.states) - 1)
        return self.states[index]


class ReasoningStateController(RetrievalQuestionController):
    def __init__(
        self,
        *,
        generator: ReasoningStateGenerator,
        max_steps: int = 2,
    ) -> None:
        super().__init__(generator=generator, max_steps=max_steps)

    @property
    def controller_type(self) -> str:
        return "reasoning_state"

    def _produce_signal(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
    ) -> SignalDecision:
        iteration = state.completed_steps + 1
        generated: str | ReasoningStateGeneration
        if iteration == 1:
            generated = payload.question
        else:
            generated = self.generator.generate(payload, retrieval_state, iteration)

        if isinstance(generated, ReasoningStateGeneration):
            reasoning_state = generated.text.strip()
            llm_calls = generated.llm_calls
            input_tokens = generated.input_tokens
            output_tokens = generated.output_tokens
            call_records = generated.call_records
            generation_fallback = generated.fallback
        else:
            reasoning_state = generated.strip()
            llm_calls = 0
            input_tokens = 0
            output_tokens = 0
            call_records = ()
            generation_fallback = False

        if iteration == 1:
            signal_text = payload.question
        elif reasoning_state:
            signal_text = compose_reasoning_signal(payload.question, reasoning_state)
        else:
            signal_text = payload.question

        return SignalDecision(
            decision_id=decision_id(self.controller_type, iteration, signal_text),
            controller_type=self.controller_type,
            iteration=iteration,
            action="retrieve",
            reason=(
                "initial retrieval from the original question"
                if iteration == 1
                else "interleaved retrieval from an evidence-grounded reasoning state"
            ),
            signal_text=signal_text,
            selected_by=type(self.generator).__name__,
            fallback=(
                generation_fallback
                or (signal_text == payload.question and iteration > 1)
            ),
            llm_calls=llm_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_records=tuple(dict(record) for record in call_records),
        )


__all__ = [
    "OpenAIReasoningStateGenerator",
    "ReasoningStateController",
    "ReasoningStateGeneration",
    "ReasoningStateGenerator",
    "StaticReasoningStateGenerator",
    "compose_reasoning_signal",
    "extract_reasoning_state",
]
