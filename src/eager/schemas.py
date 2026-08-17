from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Annotated, ClassVar, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


StableId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^\S+$")]
Name = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
QuestionHandleId = Annotated[str, StringConstraints(pattern=r"^qh_[0-9a-f]{32}$")]


FORBIDDEN_INFERENCE_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "answerable",
        "answerability",
        "decomposition",
        "gold_answer",
        "gold_relation",
        "gold_relations",
        "gold_type",
        "handle",
        "handle_id",
        "hop",
        "hop_count",
        "hops",
        "id",
        "join_key",
        "official_decomposition",
        "qid",
        "question_decomposition",
        "question_id",
        "question_type",
        "raw_dataset_id",
        "raw_id",
        "raw_question_id",
        "relation",
        "relations",
        "support",
        "support_ids",
        "supporting_facts",
        "supporting_paragraph_ids",
        "supporting_sentence_ids",
        "supporting_titles",
        "type",
    }
)


class SchemaInvariantError(ValueError):
    """Raised when a cross-record ordering or identity invariant is violated."""


class BudgetExceededError(SchemaInvariantError):
    """Raised when a state or transition exceeds its declared budget."""


class FrozenSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


T = TypeVar("T")


def _as_ordered_sequence(value: Any, *, field_name: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError(f"{field_name} must be an ordered sequence")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an ordered sequence")
    return tuple(value)


def _stable_texts(
    value: Any,
    *,
    field_name: str,
    case_insensitive: bool = False,
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in _as_ordered_sequence(value, field_name=field_name):
        cleaned = str(raw).strip()
        if not cleaned:
            raise ValueError(f"{field_name} cannot contain an empty value")
        key = cleaned.casefold() if case_insensitive else cleaned
        if key not in seen:
            seen.add(key)
            output.append(cleaned)
    return tuple(output)


def _stable_models(
    values: Sequence[T],
    *,
    key,
    field_name: str,
) -> tuple[T, ...]:
    output: list[T] = []
    seen: dict[str, T] = {}
    for item in values:
        identity = str(key(item))
        previous = seen.get(identity)
        if previous is None:
            seen[identity] = item
            output.append(item)
        elif previous != item:
            raise ValueError(f"{field_name} contains conflicting records for stable id {identity!r}")
    return tuple(output)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def assert_annotation_isolation(value: Any, *, path: str = "inference") -> None:
    """Reject annotation and identifier keys anywhere in an inference mapping."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if _normalized_key(key) in FORBIDDEN_INFERENCE_FIELDS:
                raise ValueError(f"annotation or identifier field is forbidden at {child_path}")
            assert_annotation_isolation(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_annotation_isolation(child, path=f"{path}[{index}]")


def _stable_digest(prefix: str, *parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:32]}"


def normalize_anchor_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


class QuestionHandle(FrozenSchema):
    """Opaque orchestration key. This object must never enter the inference engine."""

    handle_id: QuestionHandleId
    dataset: Name
    split: Name
    raw_dataset_id: NonEmptyText

    @field_validator("dataset", "split")
    @classmethod
    def normalize_names(cls, value: str) -> str:
        return value.lower()

    @classmethod
    def create(cls, *, dataset: str, split: str, raw_dataset_id: str) -> "QuestionHandle":
        normalized_dataset = dataset.strip().lower()
        normalized_split = split.strip().lower()
        raw_id = raw_dataset_id.strip()
        return cls(
            handle_id=_stable_digest("qh", normalized_dataset, normalized_split, raw_id),
            dataset=normalized_dataset,
            split=normalized_split,
            raw_dataset_id=raw_id,
        )


class InferencePayload(FrozenSchema):
    """The complete allowlist of fields visible to routers and controllers."""

    dataset: Name
    question: NonEmptyText

    @model_validator(mode="before")
    @classmethod
    def reject_annotation_fields(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            assert_annotation_isolation(value)
        return value

    @field_validator("dataset")
    @classmethod
    def normalize_dataset(cls, value: str) -> str:
        return value.lower()


class EvaluationDecompositionStep(FrozenSchema):
    step_id: StableId
    question: NonEmptyText
    answer: str | None = None
    depends_on: tuple[StableId, ...] = ()
    supporting_sentence_ids: tuple[StableId, ...] = ()
    supporting_paragraph_ids: tuple[StableId, ...] = ()
    relation: str | None = None

    @field_validator("depends_on", "supporting_sentence_ids", "supporting_paragraph_ids", mode="before")
    @classmethod
    def normalize_ids(cls, value: Any, info) -> tuple[str, ...]:
        return _stable_texts(value, field_name=info.field_name)

    @model_validator(mode="after")
    def reject_self_dependency(self) -> "EvaluationDecompositionStep":
        if self.step_id in self.depends_on:
            raise ValueError("a decomposition step cannot depend on itself")
        return self


class EvaluationAnnotations(FrozenSchema):
    """Evaluator-only labels joined to predictions through ``handle_id``."""

    handle_id: QuestionHandleId
    answer: str
    answer_aliases: tuple[NonEmptyText, ...] = ()
    supporting_sentence_ids: tuple[StableId, ...] = ()
    supporting_paragraph_ids: tuple[StableId, ...] = ()
    supporting_titles: tuple[NonEmptyText, ...] = ()
    gold_type: str | None = None
    hop_count: int | None = Field(default=None, ge=1)
    official_decomposition: tuple[EvaluationDecompositionStep, ...] = ()
    gold_relations: tuple[NonEmptyText, ...] = ()
    answerable: bool | None = None

    @field_validator(
        "supporting_sentence_ids",
        "supporting_paragraph_ids",
        "supporting_titles",
        "gold_relations",
        mode="before",
    )
    @classmethod
    def normalize_text_sequences(cls, value: Any, info) -> tuple[str, ...]:
        return _stable_texts(value, field_name=info.field_name)

    @field_validator("answer_aliases", mode="before")
    @classmethod
    def normalize_answer_aliases(cls, value: Any) -> tuple[str, ...]:
        return _stable_texts(value, field_name="answer_aliases", case_insensitive=True)

    @field_validator("official_decomposition", mode="after")
    @classmethod
    def unique_decomposition_steps(
        cls, values: tuple[EvaluationDecompositionStep, ...]
    ) -> tuple[EvaluationDecompositionStep, ...]:
        return _stable_models(values, key=lambda item: item.step_id, field_name="official_decomposition")


class EvidenceUnit(FrozenSchema):
    evidence_id: StableId
    paragraph_id: StableId
    title: str = ""
    sentence_id: StableId
    text: NonEmptyText


class QueryAnchor(FrozenSchema):
    anchor_id: StableId
    surface: NonEmptyText
    normalized_surface: NonEmptyText
    sources: tuple[Name, ...] = Field(min_length=1)
    priority: int = Field(ge=0)
    aliases: tuple[NonEmptyText, ...] = ()

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_sources(cls, value: Any) -> tuple[str, ...]:
        return tuple(item.lower() for item in _stable_texts(value, field_name="sources", case_insensitive=True))

    @field_validator("aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> tuple[str, ...]:
        return _stable_texts(value, field_name="aliases", case_insensitive=True)

    @field_validator("normalized_surface")
    @classmethod
    def require_normalized_surface(cls, value: str) -> str:
        normalized = normalize_anchor_surface(value)
        if value != normalized:
            raise ValueError("normalized_surface must be NFKC-normalized, case-folded, and whitespace-normalized")
        return value

    @classmethod
    def from_surface(
        cls,
        surface: str,
        *,
        sources: Sequence[str],
        priority: int,
        aliases: Sequence[str] = (),
    ) -> "QueryAnchor":
        normalized = normalize_anchor_surface(surface)
        normalized_sources = _stable_texts(sources, field_name="sources", case_insensitive=True)
        return cls(
            anchor_id=_stable_digest("qa", normalized),
            surface=surface,
            normalized_surface=normalized,
            sources=normalized_sources,
            priority=priority,
            aliases=aliases,
        )


class ScoreFeature(FrozenSchema):
    name: Name
    value: float

    @field_validator("value")
    @classmethod
    def require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score feature values must be finite")
        return value


class EvidenceCandidate(FrozenSchema):
    evidence: EvidenceUnit
    sources: tuple[Name, ...] = Field(min_length=1)
    anchor_ids: tuple[StableId, ...] = ()
    coarse_features: tuple[ScoreFeature, ...] = ()
    coarse_score: float
    fine_score: float | None = None
    coarse_rank: int | None = Field(default=None, ge=1)
    paragraph_rank: int | None = Field(default=None, ge=1)
    rank: int | None = Field(default=None, ge=1)

    @property
    def candidate_id(self) -> str:
        return self.evidence.evidence_id

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_sources(cls, value: Any) -> tuple[str, ...]:
        return tuple(item.lower() for item in _stable_texts(value, field_name="sources", case_insensitive=True))

    @field_validator("anchor_ids", mode="before")
    @classmethod
    def normalize_anchor_ids(cls, value: Any) -> tuple[str, ...]:
        return _stable_texts(value, field_name="anchor_ids")

    @field_validator("coarse_features", mode="after")
    @classmethod
    def unique_features(cls, values: tuple[ScoreFeature, ...]) -> tuple[ScoreFeature, ...]:
        return _stable_models(values, key=lambda item: item.name, field_name="coarse_features")

    @field_validator("coarse_score", "fine_score")
    @classmethod
    def finite_scores(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("candidate scores must be finite")
        return value

    @model_validator(mode="after")
    def ranked_fine_scores_exist(self) -> "EvidenceCandidate":
        if self.rank is not None and self.fine_score is None:
            raise ValueError("a final rank requires a fine_score")
        return self


class StepBudgetUsage(FrozenSchema):
    candidates_considered: int = Field(default=0, ge=0)
    parent_units_selected: int = Field(default=0, ge=0)
    rerank_pairs: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class BudgetUsage(FrozenSchema):
    iterations: int = Field(default=0, ge=0)
    candidates_considered: int = Field(default=0, ge=0)
    parent_units_selected: int = Field(default=0, ge=0)
    rerank_pairs: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    def add_step(self, step: StepBudgetUsage) -> "BudgetUsage":
        return BudgetUsage(
            iterations=self.iterations + 1,
            candidates_considered=self.candidates_considered + step.candidates_considered,
            parent_units_selected=self.parent_units_selected + step.parent_units_selected,
            rerank_pairs=self.rerank_pairs + step.rerank_pairs,
            llm_calls=self.llm_calls + step.llm_calls,
            input_tokens=self.input_tokens + step.input_tokens,
            output_tokens=self.output_tokens + step.output_tokens,
        )


class BudgetContract(FrozenSchema):
    max_iterations: int = Field(ge=0)
    max_candidates_per_step: int = Field(ge=0)
    max_candidates_per_question: int = Field(ge=0)
    max_parent_units_per_step: int = Field(ge=0)
    max_parent_units_per_question: int = Field(ge=0)
    max_rerank_pairs_per_step: int = Field(ge=0)
    max_rerank_pairs_per_question: int = Field(ge=0)
    max_retained_evidence: int = Field(ge=0)
    max_llm_calls_per_step: int = Field(ge=0)
    max_llm_calls_per_question: int = Field(ge=0)
    max_input_tokens_per_question: int = Field(ge=0)
    max_output_tokens_per_question: int = Field(ge=0)

    _STEP_TOTAL_PAIRS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("max_candidates_per_step", "max_candidates_per_question"),
        ("max_parent_units_per_step", "max_parent_units_per_question"),
        ("max_rerank_pairs_per_step", "max_rerank_pairs_per_question"),
        ("max_llm_calls_per_step", "max_llm_calls_per_question"),
    )

    @model_validator(mode="after")
    def totals_cover_one_step(self) -> "BudgetContract":
        for step_name, total_name in self._STEP_TOTAL_PAIRS:
            if getattr(self, total_name) < getattr(self, step_name):
                raise ValueError(f"{total_name} must be at least {step_name}")
        return self

    def assert_within(
        self,
        *,
        step: StepBudgetUsage,
        total: BudgetUsage,
        retained_evidence_count: int,
    ) -> None:
        checks = (
            (step.candidates_considered, self.max_candidates_per_step, "step candidates"),
            (step.parent_units_selected, self.max_parent_units_per_step, "step parent units"),
            (step.rerank_pairs, self.max_rerank_pairs_per_step, "step rerank pairs"),
            (step.llm_calls, self.max_llm_calls_per_step, "step LLM calls"),
            (total.iterations, self.max_iterations, "iterations"),
            (total.candidates_considered, self.max_candidates_per_question, "question candidates"),
            (total.parent_units_selected, self.max_parent_units_per_question, "question parent units"),
            (total.rerank_pairs, self.max_rerank_pairs_per_question, "question rerank pairs"),
            (total.llm_calls, self.max_llm_calls_per_question, "question LLM calls"),
            (total.input_tokens, self.max_input_tokens_per_question, "question input tokens"),
            (total.output_tokens, self.max_output_tokens_per_question, "question output tokens"),
            (retained_evidence_count, self.max_retained_evidence, "retained evidence"),
        )
        for actual, limit, label in checks:
            if actual > limit:
                raise BudgetExceededError(f"{label} budget exceeded: {actual} > {limit}")


SignalAction = Literal["retrieve", "stop"]


class SignalDecision(FrozenSchema):
    decision_id: StableId
    controller_type: Name
    iteration: int = Field(ge=1)
    action: SignalAction
    reason: NonEmptyText
    signal_text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    selected_by: str | None = None
    selected_step_id: StableId | None = None
    fallback: bool = False
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    call_records: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_stop_decision(self) -> "SignalDecision":
        if self.action == "stop" and self.fallback:
            raise ValueError("a stop decision cannot be marked as fallback retrieval")
        if self.signal_text is not None and not self.signal_text.strip():
            raise ValueError("signal_text must be non-empty when provided")
        return self


class GeneratedSignalStep(FrozenSchema):
    step_id: StableId
    text: NonEmptyText
    depends_on: tuple[StableId, ...] = ()

    @field_validator("depends_on", mode="before")
    @classmethod
    def normalize_dependencies(cls, value: Any) -> tuple[str, ...]:
        return _stable_texts(value, field_name="depends_on")

    @model_validator(mode="after")
    def reject_self_dependency(self) -> "GeneratedSignalStep":
        if self.step_id in self.depends_on:
            raise ValueError("a generated signal step cannot depend on itself")
        return self


class FinalOperation(FrozenSchema):
    type: Literal["return", "compare", "select", "synthesize"]
    inputs: tuple[StableId, ...] = Field(min_length=1)
    instruction: NonEmptyText

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: Any) -> tuple[str, ...]:
        return _stable_texts(value, field_name="final operation inputs")


class StepAnswer(FrozenSchema):
    step_id: StableId
    answer: NonEmptyText


ControllerPhase = Literal["ready", "awaiting_evidence", "stopped"]


class ControllerState(FrozenSchema):
    controller_type: Name
    phase: ControllerPhase = "ready"
    completed_steps: int = Field(default=0, ge=0)
    budget: BudgetContract
    generated_chain: tuple[GeneratedSignalStep, ...] = ()
    final_operation: FinalOperation | None = None
    step_answers: tuple[StepAnswer, ...] = ()
    failed_step_ids: tuple[StableId, ...] = ()
    blocked_step_ids: tuple[StableId, ...] = ()
    fallback_count: int = Field(default=0, ge=0)
    consumed_signal_ids: tuple[StableId, ...] = ()
    pending_signal_id: StableId | None = None
    pending_fallback: bool = False
    stop_reason: str | None = None

    @field_validator("consumed_signal_ids", mode="before")
    @classmethod
    def normalize_consumed_ids(cls, value: Any) -> tuple[str, ...]:
        return _stable_texts(value, field_name="consumed_signal_ids")

    @field_validator("failed_step_ids", "blocked_step_ids", mode="before")
    @classmethod
    def normalize_step_status_ids(cls, value: Any) -> tuple[str, ...]:
        return _stable_texts(value, field_name="step status ids")

    @field_validator("generated_chain", mode="after")
    @classmethod
    def unique_chain_steps(cls, values: tuple[GeneratedSignalStep, ...]) -> tuple[GeneratedSignalStep, ...]:
        unique = _stable_models(values, key=lambda item: item.step_id, field_name="generated_chain")
        prior: set[str] = set()
        for step in unique:
            missing = set(step.depends_on) - prior
            if missing:
                raise ValueError(f"generated step {step.step_id!r} depends on non-prior steps: {sorted(missing)}")
            prior.add(step.step_id)
        return unique

    @field_validator("step_answers", mode="after")
    @classmethod
    def unique_step_answers(cls, values: tuple[StepAnswer, ...]) -> tuple[StepAnswer, ...]:
        return _stable_models(values, key=lambda item: item.step_id, field_name="step_answers")

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "ControllerState":
        if self.completed_steps > self.budget.max_iterations:
            raise BudgetExceededError("controller completed_steps exceeds max_iterations")
        if len(self.consumed_signal_ids) != self.completed_steps:
            raise ValueError("each completed controller step must consume exactly one signal")
        if self.fallback_count > self.completed_steps:
            raise ValueError("fallback_count cannot exceed completed_steps")
        if len(self.step_answers) > self.completed_steps:
            raise ValueError("step_answers cannot exceed completed_steps")
        failed = set(self.failed_step_ids)
        blocked = set(self.blocked_step_ids)
        chain_ids = {step.step_id for step in self.generated_chain}
        if not failed <= chain_ids or not blocked <= chain_ids:
            raise ValueError("step status ids must reference generated chain steps")
        if failed & blocked:
            raise ValueError("a step cannot be both failed and blocked")
        if self.phase == "awaiting_evidence":
            if self.pending_signal_id is None or self.stop_reason is not None:
                raise ValueError("awaiting_evidence requires one pending signal and no stop reason")
        elif self.phase == "ready":
            if self.pending_signal_id is not None or self.pending_fallback or self.stop_reason is not None:
                raise ValueError("ready state cannot retain pending or stop fields")
        else:
            if self.pending_signal_id is not None or self.pending_fallback or not self.stop_reason:
                raise ValueError("stopped state requires only a non-empty stop reason")
        return self

    @classmethod
    def started(
        cls,
        *,
        controller_type: str,
        budget: BudgetContract,
        generated_chain: Sequence[GeneratedSignalStep] = (),
        final_operation: FinalOperation | None = None,
    ) -> "ControllerState":
        return cls(
            controller_type=controller_type,
            budget=budget,
            generated_chain=tuple(generated_chain),
            final_operation=final_operation,
        )

    def begin_signal(self, decision: SignalDecision) -> "ControllerState":
        if self.phase != "ready":
            raise SchemaInvariantError("a controller can produce a signal only from ready state")
        if self.completed_steps >= self.budget.max_iterations:
            raise BudgetExceededError("controller iteration budget is exhausted")
        if decision.controller_type != self.controller_type:
            raise SchemaInvariantError("signal controller_type does not match controller state")
        if decision.iteration != self.completed_steps + 1:
            raise SchemaInvariantError("signal iteration must be the next controller iteration")
        if decision.action == "stop":
            return ControllerState.model_validate(
                {
                    **self.model_dump(mode="python"),
                    "phase": "stopped",
                    "stop_reason": decision.reason,
                }
            )
        return ControllerState.model_validate(
            {
                **self.model_dump(mode="python"),
                "phase": "awaiting_evidence",
                "pending_signal_id": decision.decision_id,
                "pending_fallback": decision.fallback,
            }
        )

    def complete_step(
        self,
        *,
        decision_id: str,
        next_action: Literal["continue", "stop"],
        reason: str,
        intermediate_answer: str | None = None,
        answer_step_id: str | None = None,
        failed_step_id: str | None = None,
        blocked_step_ids: Sequence[str] = (),
    ) -> "ControllerState":
        if self.phase != "awaiting_evidence" or self.pending_signal_id != decision_id:
            raise SchemaInvariantError("only the pending signal can be consumed")
        completed = self.completed_steps + 1
        if next_action == "continue" and completed >= self.budget.max_iterations:
            raise BudgetExceededError("controller must stop after consuming its final allowed iteration")
        answers = list(self.step_answers)
        if (intermediate_answer is None) != (answer_step_id is None):
            raise SchemaInvariantError("intermediate_answer and answer_step_id must be provided together")
        if intermediate_answer is not None and answer_step_id is not None:
            answers.append(StepAnswer(step_id=answer_step_id, answer=intermediate_answer))
        failed = list(self.failed_step_ids)
        blocked = list(self.blocked_step_ids)
        if failed_step_id is not None and failed_step_id not in failed:
            failed.append(failed_step_id)
        for step_id in blocked_step_ids:
            if step_id not in blocked:
                blocked.append(step_id)
        return ControllerState.model_validate(
            {
                **self.model_dump(mode="python"),
                "phase": "ready" if next_action == "continue" else "stopped",
                "completed_steps": completed,
                "step_answers": answers,
                "failed_step_ids": failed,
                "blocked_step_ids": blocked,
                "fallback_count": self.fallback_count + int(self.pending_fallback),
                "consumed_signal_ids": (*self.consumed_signal_ids, decision_id),
                "pending_signal_id": None,
                "pending_fallback": False,
                "stop_reason": None if next_action == "continue" else reason,
            }
        )


class StepOutcome(FrozenSchema):
    decision_id: StableId
    iteration: int = Field(ge=1)
    consumed_evidence_ids: tuple[StableId, ...] = ()
    intermediate_answer: str | None = None
    answer_step_id: StableId | None = None
    step_status: Literal["succeeded", "retrying", "failed", "blocked"] = "succeeded"
    next_action: Literal["continue", "stop"]
    reason: NonEmptyText
    used_fallback: bool = False
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    call_records: tuple[dict[str, Any], ...] = ()
    state: ControllerState

    @field_validator("consumed_evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> tuple[str, ...]:
        return _stable_texts(value, field_name="consumed_evidence_ids")

    @model_validator(mode="after")
    def validate_transition(self) -> "StepOutcome":
        if (self.intermediate_answer is None) != (self.answer_step_id is None):
            raise ValueError("intermediate_answer and answer_step_id must be provided together")
        if self.state.completed_steps != self.iteration:
            raise ValueError("outcome iteration must equal completed controller steps")
        if not self.state.consumed_signal_ids or self.state.consumed_signal_ids[-1] != self.decision_id:
            raise ValueError("outcome decision must be the latest consumed signal")
        expected_phase = "ready" if self.next_action == "continue" else "stopped"
        if self.state.phase != expected_phase:
            raise ValueError("outcome next_action does not match controller phase")
        return self

    @classmethod
    def create(
        cls,
        *,
        state: ControllerState,
        decision: SignalDecision,
        consumed_evidence_ids: Sequence[str] = (),
        intermediate_answer: str | None = None,
        answer_step_id: str | None = None,
        step_status: Literal["succeeded", "retrying", "failed", "blocked"] = "succeeded",
        failed_step_id: str | None = None,
        blocked_step_ids: Sequence[str] = (),
        next_action: Literal["continue", "stop"],
        reason: str,
        llm_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_records: Sequence[Mapping[str, Any]] = (),
    ) -> "StepOutcome":
        next_state = state.complete_step(
            decision_id=decision.decision_id,
            next_action=next_action,
            reason=reason,
            intermediate_answer=intermediate_answer,
            answer_step_id=answer_step_id,
            failed_step_id=failed_step_id,
            blocked_step_ids=blocked_step_ids,
        )
        return cls(
            decision_id=decision.decision_id,
            iteration=next_state.completed_steps,
            consumed_evidence_ids=tuple(consumed_evidence_ids),
            intermediate_answer=intermediate_answer,
            answer_step_id=answer_step_id,
            next_action=next_action,
            reason=reason,
            step_status=step_status,
            used_fallback=state.pending_fallback,
            llm_calls=llm_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_records=tuple(dict(record) for record in call_records),
            state=next_state,
        )


class RetrievalState(FrozenSchema):
    iteration: int = Field(default=0, ge=0)
    anchors: tuple[QueryAnchor, ...] = ()
    candidate_pool: tuple[EvidenceCandidate, ...] = ()
    retained_evidence: tuple[EvidenceUnit, ...] = ()
    # Fine-ranked evidence from the most recent retrieval step. It is a
    # working context for intermediate answers, not part of final evidence.
    current_evidence: tuple[EvidenceUnit, ...] = ()
    signal_history: tuple[SignalDecision, ...] = ()
    budget: BudgetContract
    step_usage: StepBudgetUsage = Field(default_factory=StepBudgetUsage)
    total_usage: BudgetUsage = Field(default_factory=BudgetUsage)

    @field_validator("anchors", mode="after")
    @classmethod
    def stable_anchor_order(cls, values: tuple[QueryAnchor, ...]) -> tuple[QueryAnchor, ...]:
        return _stable_models(values, key=lambda item: item.anchor_id, field_name="anchors")

    @field_validator("candidate_pool", mode="after")
    @classmethod
    def stable_candidate_order(
        cls, values: tuple[EvidenceCandidate, ...]
    ) -> tuple[EvidenceCandidate, ...]:
        return _stable_models(values, key=lambda item: item.candidate_id, field_name="candidate_pool")

    @field_validator("retained_evidence", mode="after")
    @classmethod
    def stable_evidence_order(cls, values: tuple[EvidenceUnit, ...]) -> tuple[EvidenceUnit, ...]:
        return _stable_models(values, key=lambda item: item.evidence_id, field_name="retained_evidence")

    @field_validator("signal_history", mode="after")
    @classmethod
    def stable_signal_order(cls, values: tuple[SignalDecision, ...]) -> tuple[SignalDecision, ...]:
        return _stable_models(values, key=lambda item: item.decision_id, field_name="signal_history")

    @model_validator(mode="after")
    def enforce_budget_and_iteration(self) -> "RetrievalState":
        if self.total_usage.iterations != self.iteration:
            raise ValueError("retrieval iteration must equal total_usage.iterations")
        if len(self.candidate_pool) > self.step_usage.candidates_considered:
            raise ValueError("candidate_pool cannot exceed the recorded candidates considered")
        self.budget.assert_within(
            step=self.step_usage,
            total=self.total_usage,
            retained_evidence_count=len(self.retained_evidence),
        )
        return self

    @classmethod
    def initial(cls, budget: BudgetContract) -> "RetrievalState":
        return cls(budget=budget)

    def advance(
        self,
        *,
        signal: SignalDecision,
        anchors: Sequence[QueryAnchor],
        candidates: Sequence[EvidenceCandidate],
        new_evidence: Sequence[EvidenceUnit],
        current_evidence: Sequence[EvidenceUnit] | None = None,
        step_usage: StepBudgetUsage,
    ) -> "RetrievalState":
        if signal.action != "retrieve":
            raise SchemaInvariantError("retrieval state can advance only for a retrieve decision")
        if signal.iteration != self.iteration + 1:
            raise SchemaInvariantError("retrieval signal iteration must be contiguous")
        merged_evidence = _stable_models(
            (*self.retained_evidence, *new_evidence),
            key=lambda item: item.evidence_id,
            field_name="retained_evidence",
        )[: self.budget.max_retained_evidence]
        next_total = self.total_usage.add_step(step_usage)
        self.budget.assert_within(
            step=step_usage,
            total=next_total,
            retained_evidence_count=len(merged_evidence),
        )
        return RetrievalState(
            iteration=self.iteration + 1,
            anchors=tuple(anchors),
            candidate_pool=tuple(candidates),
            retained_evidence=merged_evidence,
            current_evidence=tuple(
                new_evidence if current_evidence is None else current_evidence
            ),
            signal_history=(*self.signal_history, signal),
            budget=self.budget,
            step_usage=step_usage,
            total_usage=next_total,
        )

    def add_current_step_usage(self, additional: StepBudgetUsage) -> "RetrievalState":
        if self.iteration == 0:
            raise SchemaInvariantError("cannot add usage before the first retrieval step")
        step_usage = StepBudgetUsage(
            candidates_considered=(
                self.step_usage.candidates_considered
                + additional.candidates_considered
            ),
            parent_units_selected=(
                self.step_usage.parent_units_selected
                + additional.parent_units_selected
            ),
            rerank_pairs=self.step_usage.rerank_pairs + additional.rerank_pairs,
            llm_calls=self.step_usage.llm_calls + additional.llm_calls,
            input_tokens=self.step_usage.input_tokens + additional.input_tokens,
            output_tokens=self.step_usage.output_tokens + additional.output_tokens,
        )
        total_usage = BudgetUsage(
            iterations=self.total_usage.iterations,
            candidates_considered=(
                self.total_usage.candidates_considered
                + additional.candidates_considered
            ),
            parent_units_selected=(
                self.total_usage.parent_units_selected
                + additional.parent_units_selected
            ),
            rerank_pairs=self.total_usage.rerank_pairs + additional.rerank_pairs,
            llm_calls=self.total_usage.llm_calls + additional.llm_calls,
            input_tokens=self.total_usage.input_tokens + additional.input_tokens,
            output_tokens=self.total_usage.output_tokens + additional.output_tokens,
        )
        self.budget.assert_within(
            step=step_usage,
            total=total_usage,
            retained_evidence_count=len(self.retained_evidence),
        )
        return self.model_copy(
            update={"step_usage": step_usage, "total_usage": total_usage}
        )

    def finalize_evidence(
        self,
        evidence: Sequence[EvidenceUnit],
        *,
        rerank_pairs: int,
    ) -> "RetrievalState":
        if rerank_pairs < 0:
            raise ValueError("final rerank_pairs must be non-negative")
        final_evidence = _stable_models(
            tuple(evidence),
            key=lambda item: item.evidence_id,
            field_name="retained_evidence",
        )[: self.budget.max_retained_evidence]
        total_usage = BudgetUsage(
            iterations=self.total_usage.iterations,
            candidates_considered=self.total_usage.candidates_considered,
            parent_units_selected=self.total_usage.parent_units_selected,
            rerank_pairs=self.total_usage.rerank_pairs + rerank_pairs,
            llm_calls=self.total_usage.llm_calls,
            input_tokens=self.total_usage.input_tokens,
            output_tokens=self.total_usage.output_tokens,
        )
        self.budget.assert_within(
            step=self.step_usage,
            total=total_usage,
            retained_evidence_count=len(final_evidence),
        )
        return RetrievalState(
            iteration=self.iteration,
            anchors=self.anchors,
            candidate_pool=self.candidate_pool,
            retained_evidence=final_evidence,
            current_evidence=self.current_evidence,
            signal_history=self.signal_history,
            budget=self.budget,
            step_usage=self.step_usage,
            total_usage=total_usage,
        )


class PredictionError(FrozenSchema):
    code: Name
    message: NonEmptyText
    retryable: bool = False


class PredictionRecord(FrozenSchema):
    handle_id: QuestionHandleId
    controller_type: Name
    status: Literal["ok", "error"]
    answer: str | None = None
    evidence: tuple[EvidenceUnit, ...] = ()
    signal_trace: tuple[SignalDecision, ...] = ()
    step_trace: tuple[StepOutcome, ...] = ()
    budget: BudgetContract
    usage: BudgetUsage
    latency_ms: float = Field(ge=0.0)
    error: PredictionError | None = None

    @field_validator("evidence", mode="after")
    @classmethod
    def stable_evidence(cls, values: tuple[EvidenceUnit, ...]) -> tuple[EvidenceUnit, ...]:
        return _stable_models(values, key=lambda item: item.evidence_id, field_name="evidence")

    @field_validator("signal_trace", mode="after")
    @classmethod
    def stable_signals(cls, values: tuple[SignalDecision, ...]) -> tuple[SignalDecision, ...]:
        return _stable_models(values, key=lambda item: item.decision_id, field_name="signal_trace")

    @field_validator("step_trace", mode="after")
    @classmethod
    def stable_steps(cls, values: tuple[StepOutcome, ...]) -> tuple[StepOutcome, ...]:
        return _stable_models(values, key=lambda item: item.decision_id, field_name="step_trace")

    @model_validator(mode="after")
    def validate_result(self) -> "PredictionRecord":
        if not math.isfinite(self.latency_ms):
            raise ValueError("latency_ms must be finite")
        if self.status == "ok":
            if self.answer is None or not self.answer.strip() or self.error is not None:
                raise ValueError("successful predictions require an answer and no error")
        elif self.answer is not None or self.error is None:
            raise ValueError("failed predictions require no answer and a structured error")
        self.budget.assert_within(
            step=StepBudgetUsage(),
            total=self.usage,
            retained_evidence_count=len(self.evidence),
        )
        return self


__all__ = [
    "BudgetContract",
    "BudgetExceededError",
    "BudgetUsage",
    "ControllerState",
    "EvaluationAnnotations",
    "EvaluationDecompositionStep",
    "EvidenceCandidate",
    "EvidenceUnit",
    "FORBIDDEN_INFERENCE_FIELDS",
    "GeneratedSignalStep",
    "InferencePayload",
    "PredictionError",
    "PredictionRecord",
    "QueryAnchor",
    "QuestionHandle",
    "RetrievalState",
    "SchemaInvariantError",
    "ScoreFeature",
    "SignalDecision",
    "StepAnswer",
    "StepBudgetUsage",
    "StepOutcome",
    "assert_annotation_isolation",
    "normalize_anchor_surface",
]
