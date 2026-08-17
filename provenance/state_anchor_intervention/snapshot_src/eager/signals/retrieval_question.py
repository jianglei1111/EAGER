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
    BudgetContract,
    ControllerState,
    EvidenceUnit,
    InferencePayload,
    RetrievalState,
    SignalDecision,
    StepOutcome,
)
from eager.signals._identity import decision_id
from eager.signals.base import SignalController


class RetrievalQuestionGenerator(Protocol):
    def generate(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        iteration: int,
    ) -> str | "RetrievalQuestionGeneration": ...


@dataclass(frozen=True, slots=True)
class RetrievalQuestionGeneration:
    text: str
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    call_records: tuple[Mapping[str, Any], ...] = ()
    fallback: bool = False


class OpenAIRetrievalQuestionGenerator:
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
            raise ValueError("retrieval-question system prompt must be non-empty")
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
            for item in self._prompt_evidence(retrieval_state)
        ]
        request = {
            "original_question": payload.question,
            "iteration": iteration,
            "previous_retrieval_questions": [
                decision.signal_text
                for decision in retrieval_state.signal_history
                if decision.signal_text
            ],
            "current_evidence": evidence,
        }
        return json.dumps(request, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _parse_text(raw_output: str) -> str:
        text = raw_output.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return ""
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ""
        if not isinstance(payload, dict):
            return ""
        return re.sub(
            r"\s+",
            " ",
            str(payload.get("retrieval_question") or "").strip(),
        )

    def generate(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        iteration: int,
    ) -> RetrievalQuestionGeneration:
        user_prompt = self._user_prompt(payload, retrieval_state, iteration)
        response = self.client.complete(
            (
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ),
            temperature=0.0,
            max_completion_tokens=400,
        )
        generated = self._parse_text(response.content)
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
        return RetrievalQuestionGeneration(
            text=generated,
            llm_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_records=(
                {
                    "kind": "retrieval_question_generation",
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
                },
            ),
            fallback=not generated,
        )


class EvidenceAwareQuestionGenerator:
    """Offline fallback generator derived only from question text and retained evidence."""

    def generate(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        iteration: int,
    ) -> str:
        if iteration == 1 or not retrieval_state.retained_evidence:
            return payload.question
        titles = tuple(
            dict.fromkeys(
                evidence.title
                for evidence in retrieval_state.retained_evidence
                if evidence.title
            )
        )[:3]
        return (
            payload.question
            if not titles
            else f"{payload.question} evidence path through: {', '.join(titles)}"
        )


@dataclass(frozen=True, slots=True)
class StaticQuestionGenerator:
    questions: tuple[str, ...]

    def __init__(self, questions: Sequence[str]) -> None:
        cleaned = tuple(question.strip() for question in questions)
        if not cleaned or any(not question for question in cleaned):
            raise ValueError("static retrieval questions must be non-empty")
        object.__setattr__(self, "questions", cleaned)

    def generate(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        iteration: int,
    ) -> str:
        index = min(iteration - 1, len(self.questions) - 1)
        return self.questions[index]


class RetrievalQuestionController(SignalController):
    def __init__(
        self,
        *,
        generator: RetrievalQuestionGenerator | None = None,
        max_steps: int = 2,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.generator = generator or EvidenceAwareQuestionGenerator()
        self.max_steps = max_steps

    @property
    def controller_type(self) -> str:
        return "retrieval_question"

    def _start(self, payload: InferencePayload, budget: BudgetContract) -> ControllerState:
        return ControllerState.started(controller_type=self.controller_type, budget=budget)

    def _produce_signal(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
    ) -> SignalDecision:
        iteration = state.completed_steps + 1
        generated: str | RetrievalQuestionGeneration
        if iteration == 1:
            generated = payload.question
        else:
            generated = self.generator.generate(payload, retrieval_state, iteration)
        if isinstance(generated, RetrievalQuestionGeneration):
            signal_text = generated.text.strip()
            llm_calls = generated.llm_calls
            input_tokens = generated.input_tokens
            output_tokens = generated.output_tokens
            call_records = generated.call_records
            generation_fallback = generated.fallback
        else:
            signal_text = generated.strip()
            llm_calls = 0
            input_tokens = 0
            output_tokens = 0
            call_records = ()
            generation_fallback = False
        if not signal_text:
            signal_text = payload.question
        return SignalDecision(
            decision_id=decision_id(self.controller_type, iteration, signal_text),
            controller_type=self.controller_type,
            iteration=iteration,
            action="retrieve",
            reason=(
                "initial retrieval from the original question"
                if iteration == 1
                else "supplement retrieval from inference-visible evidence state"
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

    def _consume_evidence(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
        decision: SignalDecision,
    ) -> StepOutcome:
        retained_ids = {
            evidence.evidence_id for evidence in retrieval_state.retained_evidence
        }
        consumed = tuple(
            candidate.evidence.evidence_id
            for candidate in retrieval_state.candidate_pool
            if candidate.evidence.evidence_id in retained_ids
        )
        exhausted = decision.iteration >= min(self.max_steps, state.budget.max_iterations)
        empty = not retrieval_state.candidate_pool
        next_action = "stop" if exhausted or empty else "continue"
        reason = (
            "no candidates retained"
            if empty
            else f"{self.controller_type.replace('_', '-')} step limit reached"
            if exhausted
            else "supplementary retrieval remains available"
        )
        return StepOutcome.create(
            state=state,
            decision=decision,
            consumed_evidence_ids=consumed,
            next_action=next_action,
            reason=reason,
        )


__all__ = [
    "EvidenceAwareQuestionGenerator",
    "OpenAIRetrievalQuestionGenerator",
    "RetrievalQuestionController",
    "RetrievalQuestionGeneration",
    "RetrievalQuestionGenerator",
    "StaticQuestionGenerator",
]
