from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from eager.schemas import EvidenceUnit, FinalOperation, InferencePayload, StepAnswer


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    payload: InferencePayload
    evidence: tuple[EvidenceUnit, ...]
    final_operation: FinalOperation | None
    step_answers: tuple[StepAnswer, ...]
    question_override: str | None

    def __init__(
        self,
        *,
        payload: InferencePayload,
        evidence: Sequence[EvidenceUnit],
        final_operation: FinalOperation | None = None,
        step_answers: Sequence[StepAnswer] = (),
        question_override: str | None = None,
    ) -> None:
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "evidence", tuple(evidence))
        object.__setattr__(self, "final_operation", final_operation)
        object.__setattr__(self, "step_answers", tuple(step_answers))
        normalized_question = str(question_override or "").strip()
        object.__setattr__(
            self,
            "question_override",
            normalized_question or None,
        )


class AnswerGenerator(Protocol):
    def generate(self, request: AnswerRequest) -> str: ...


class AnswerGenerationError(ValueError):
    """Answer failure that preserves audited usage without storing response text."""

    def __init__(
        self,
        message: str,
        *,
        llm_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_records: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.llm_calls = llm_calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.call_records = tuple(call_records)


@dataclass(frozen=True, slots=True)
class AnswerGeneration:
    answer: str
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    call_record: Mapping[str, Any] | None = None
    call_records: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError("answer generation must be non-empty")
        for field in ("llm_calls", "input_tokens", "output_tokens"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")


class AuditedAnswerGenerator(Protocol):
    def generate_audited(self, request: AnswerRequest) -> AnswerGeneration: ...


@dataclass(frozen=True, slots=True)
class StaticAnswerGenerator:
    answer: str

    def generate(self, request: AnswerRequest) -> str:
        return self.answer.strip()


class CallableAnswerGenerator:
    """Adapter for an audited LLM or local answer implementation."""

    def __init__(self, generate: Callable[[AnswerRequest], str]) -> None:
        self._generate = generate

    def generate(self, request: AnswerRequest) -> str:
        return self._generate(request).strip()


__all__ = [
    "AnswerGeneration",
    "AnswerGenerationError",
    "AnswerGenerator",
    "AnswerRequest",
    "AuditedAnswerGenerator",
    "CallableAnswerGenerator",
    "StaticAnswerGenerator",
]
