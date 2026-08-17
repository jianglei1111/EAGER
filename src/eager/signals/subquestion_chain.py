from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from eager.answering import AnswerGenerationError, AnswerRequest, OpenAIAnswerGenerator
from eager.config import canonical_json_bytes
from eager.runtime.llm_client import ChatMessage, OpenAICompatibleClient
from eager.schemas import (
    BudgetContract,
    ControllerState,
    FinalOperation,
    GeneratedSignalStep,
    InferencePayload,
    RetrievalState,
    SignalDecision,
    StepOutcome,
)
from eager.signals._identity import decision_id
from eager.signals.base import SignalController


_NUMBERED_PLACEHOLDER = re.compile(r"#(\d+)")
_TEXTUAL_STEP_REFERENCE = re.compile(r"\bstep\s*[_#-]?\s*(\d+)\b", re.IGNORECASE)

_MINIMAL_PLAN_REPAIR_INSTRUCTION = (
    "This is a MINIMAL EDIT task, not a new planning task. Treat "
    "previous_invalid_plan as the source of truth. Preserve every existing "
    "step id, question meaning, dependency relation, answer-changing constraint, "
    "branch, and final-operation intent. Do not regenerate, shorten, reorder, "
    "or replace the decomposition. Apply only the smallest JSON edit required "
    "by validation_error. Use this mechanical repair order: (1) for every "
    "step, make the set of #n placeholders exactly equal to depends_on without "
    "dropping a dependency; (2) compute the transitive dependency closure of "
    "final_operation.inputs and make it cover every retained step; for an "
    "orphan branch, add that branch's terminal step to inputs, and if a return "
    "operation needs multiple branch outputs, change only its type to synthesize "
    "and keep all terminal inputs; (3) fix final-operation arity and references; "
    "(4) fix ids only by minimal renumbering. Never delete a branch or invent a "
    "new fact. Keep the same JSON schema key (steps or subquestions). Use no "
    "answers, evidence, gold labels, hop labels, or outside knowledge. Return "
    "the full minimally edited JSON object and no prose."
)


class ChainGenerator(Protocol):
    def generate(
        self,
        payload: InferencePayload,
        max_steps: int,
    ) -> Sequence[GeneratedSignalStep] | "ChainGeneration": ...


class IntermediateAnswerer(Protocol):
    def answer(
        self,
        payload: InferencePayload,
        step: GeneratedSignalStep,
        retrieval_state: RetrievalState,
    ) -> str | "IntermediateAnswerGeneration": ...


@dataclass(frozen=True, slots=True)
class ChainGeneration:
    steps: tuple[GeneratedSignalStep, ...]
    final_operation: FinalOperation | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    call_records: tuple[Mapping[str, Any], ...] = ()
    fallback: bool = False


@dataclass(frozen=True, slots=True)
class IntermediateAnswerGeneration:
    answer: str
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    call_records: tuple[Mapping[str, Any], ...] = ()
    fallback: bool = False


def _json_object(raw_output: str) -> dict[str, Any] | None:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class OpenAIChainGenerator:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        system_prompt: str,
        max_attempts: int = 2,
        review_valid_plan: bool = False,
    ) -> None:
        prompt = system_prompt.strip()
        if not prompt:
            raise ValueError("subquestion-chain system prompt must be non-empty")
        if max_attempts < 1:
            raise ValueError("subquestion-chain max_attempts must be positive")
        self.client = client
        self.system_prompt = prompt
        self.system_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.max_attempts = max_attempts
        self.review_valid_plan = review_valid_plan

    @staticmethod
    def _parse_plan_with_error(
        raw_output: str,
        max_steps: int,
    ) -> tuple[tuple[GeneratedSignalStep, ...], FinalOperation | None, str | None]:
        parsed = _json_object(raw_output)
        dag_plan = parsed is not None and "steps" in parsed
        rows = parsed.get("steps" if dag_plan else "subquestions") if parsed is not None else None
        if not isinstance(rows, list) or not rows:
            return (), None, "steps must be a non-empty list"
        if len(rows) > max_steps:
            return (), None, f"plan has {len(rows)} steps but the inference budget allows {max_steps}"
        steps: list[GeneratedSignalStep] = []
        seen_ids: set[int] = set()
        try:
            for ordinal, row in enumerate(rows, start=1):
                if not isinstance(row, Mapping):
                    return (), None, f"step {ordinal} must be an object"
                raw_id = int(row.get("id"))
                question = re.sub(r"\s+", " ", str(row.get("question") or "")).strip()
                raw_dependencies = row.get("depends_on")
                if (
                    raw_id != ordinal
                    or raw_id in seen_ids
                    or not question
                ):
                    return (), None, (
                        f"step {ordinal} must use id={ordinal} and a non-empty question"
                    )
                placeholder_ids = tuple(
                    int(value) for value in _NUMBERED_PLACEHOLDER.findall(question)
                )
                textual_references = tuple(
                    int(value) for value in _TEXTUAL_STEP_REFERENCE.findall(question)
                )
                if textual_references:
                    return (), None, (
                        f"step {ordinal} must use #n placeholders instead of textual "
                        "step references"
                    )
                if raw_dependencies is None:
                    dependency_ids = tuple(dict.fromkeys(placeholder_ids))
                else:
                    if not isinstance(raw_dependencies, list):
                        return (), None, f"step {ordinal} dependencies must be a list"
                    dependency_ids = tuple(int(value) for value in raw_dependencies)
                    if len(dependency_ids) != len(set(dependency_ids)):
                        return (), None, f"step {ordinal} contains duplicate dependencies"
                    placeholder_set = set(placeholder_ids)
                    dependency_set = set(dependency_ids)
                    if placeholder_set != dependency_set:
                        missing_placeholders = sorted(dependency_set - placeholder_set)
                        undeclared_placeholders = sorted(placeholder_set - dependency_set)
                        return (), None, (
                            f"step {ordinal} dependency placeholders do not match "
                            f"depends_on; missing={missing_placeholders}, "
                            f"undeclared={undeclared_placeholders}"
                        )
                if any(value not in seen_ids for value in dependency_ids):
                    return (), None, f"step {ordinal} depends on a non-prior step"
                steps.append(
                    GeneratedSignalStep(
                        step_id=f"step_{raw_id}",
                        text=question,
                        depends_on=tuple(f"step_{value}" for value in dependency_ids),
                    )
                )
                seen_ids.add(raw_id)
        except (TypeError, ValueError):
            return (), None, "step ids and dependencies must be integers"

        final_operation: FinalOperation | None = None
        if dag_plan:
            raw_final = parsed.get("final_operation")
            if raw_final is None:
                return tuple(steps), None, None
            if not isinstance(raw_final, Mapping):
                return (), None, "final_operation must be an object"
            raw_inputs = raw_final.get("inputs")
            if not isinstance(raw_inputs, list) or not raw_inputs:
                return (), None, "final_operation.inputs must be a non-empty integer list"
            try:
                input_ids = tuple(int(value) for value in raw_inputs)
            except (TypeError, ValueError):
                return (), None, "final_operation.inputs must contain integers"
            if len(input_ids) != len(set(input_ids)) or any(value not in seen_ids for value in input_ids):
                return (), None, "final_operation.inputs must reference unique existing steps"
            try:
                final_operation = FinalOperation(
                    type=raw_final.get("type"),
                    inputs=tuple(f"step_{value}" for value in input_ids),
                    instruction=str(raw_final.get("instruction") or "").strip(),
                )
            except (TypeError, ValueError) as exc:
                return (), None, f"invalid final_operation: {exc}"
            if final_operation.type == "return" and len(final_operation.inputs) != 1:
                return (), None, "return final_operation must have exactly one input"
            if final_operation.type in {"compare", "select"} and len(final_operation.inputs) < 2:
                return (), None, (
                    f"{final_operation.type} final_operation must have at least two inputs"
                )

            ancestors = set(final_operation.inputs)
            frontier = list(final_operation.inputs)
            by_id = {step.step_id: step for step in steps}
            while frontier:
                current = frontier.pop()
                for dependency in by_id[current].depends_on:
                    if dependency not in ancestors:
                        ancestors.add(dependency)
                        frontier.append(dependency)
            expected = {step.step_id for step in steps}
            if ancestors != expected:
                missing = sorted(expected - ancestors)
                return (), None, f"final_operation leaves orphan steps: {missing}"
            return tuple(steps), final_operation, None

        if len(steps) > 1 and not steps[-1].depends_on:
            return (), None, "the final subquestion must depend on the preceding reasoning chain"
        ancestors = set(steps[-1].depends_on)
        frontier = list(steps[-1].depends_on)
        by_id = {step.step_id: step for step in steps}
        while frontier:
            current = frontier.pop()
            for dependency in by_id[current].depends_on:
                if dependency not in ancestors:
                    ancestors.add(dependency)
                    frontier.append(dependency)
        expected_ancestors = {step.step_id for step in steps[:-1]}
        if ancestors != expected_ancestors:
            missing = sorted(expected_ancestors - ancestors)
            return (), None, f"final subquestion does not depend on earlier steps: {missing}"
        return tuple(steps), None, None

    @staticmethod
    def _parse_steps_with_error(
        raw_output: str,
        max_steps: int,
    ) -> tuple[tuple[GeneratedSignalStep, ...], str | None]:
        steps, _final_operation, error = OpenAIChainGenerator._parse_plan_with_error(
            raw_output,
            max_steps,
        )
        return steps, error

    @staticmethod
    def _parse_steps(raw_output: str, max_steps: int) -> tuple[GeneratedSignalStep, ...]:
        steps, _ = OpenAIChainGenerator._parse_steps_with_error(raw_output, max_steps)
        return steps

    def generate(
        self,
        payload: InferencePayload,
        max_steps: int,
    ) -> ChainGeneration:
        steps: tuple[GeneratedSignalStep, ...] = ()
        final_operation: FinalOperation | None = None
        validation_error: str | None = None
        draft_steps: tuple[GeneratedSignalStep, ...] = ()
        draft_final_operation: FinalOperation | None = None
        previous_plan: dict[str, Any] | None = None
        call_records: list[dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        for plan_attempt in range(1, self.max_attempts + 1):
            request: dict[str, Any] = {"original_question": payload.question}
            reviewing = bool(draft_steps)
            if reviewing:
                request["draft_plan"] = previous_plan
                request["review_instruction"] = (
                    "Audit the draft against the original question only. Reconstruct the "
                    "hidden relation chain, add any omitted atomic bridge lookup, preserve "
                    "all necessary constraints, and return the complete corrected JSON plan. "
                    "Do not shorten a dependency-complete draft and do not use dataset ids, "
                    "hop labels, answers, or evidence."
                )
            elif validation_error is not None:
                request["previous_invalid_plan"] = previous_plan
                request["retry_instruction"] = (
                    _MINIMAL_PLAN_REPAIR_INSTRUCTION
                    + " The parser reported: "
                    + validation_error
                )
                request["validation_error"] = validation_error
            user_prompt = json.dumps(
                request,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            response = self.client.complete(
                (
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ),
                temperature=0.0,
                max_completion_tokens=900,
            )
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            candidate_steps, candidate_operation, candidate_error = self._parse_plan_with_error(
                response.content,
                max_steps,
            )
            candidate_plan = _json_object(response.content)
            accepted = False
            if candidate_steps:
                if reviewing:
                    if len(candidate_steps) >= len(draft_steps):
                        steps = candidate_steps
                        final_operation = candidate_operation
                        previous_plan = candidate_plan
                        accepted = True
                    else:
                        steps = draft_steps
                        final_operation = draft_final_operation
                else:
                    steps = candidate_steps
                    final_operation = candidate_operation
                    previous_plan = candidate_plan
                    accepted = True
            elif not reviewing:
                steps = ()
                previous_plan = candidate_plan
            validation_error = candidate_error
            request_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "system_prompt_sha256": self.system_prompt_sha256,
                        "user_prompt": user_prompt,
                    }
                )
            ).hexdigest()
            call_records.append(
                {
                    "kind": "subquestion_decomposition",
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
                    "generated_step_count": len(candidate_steps),
                    "final_operation_type": (
                        candidate_operation.type if candidate_operation is not None else None
                    ),
                    "plan_attempt": plan_attempt,
                    "plan_review": reviewing,
                    "plan_accepted": accepted,
                    "validation_error": candidate_error,
                    "fallback": not (steps or draft_steps),
                }
            )
            if reviewing:
                break
            if candidate_steps and self.review_valid_plan and plan_attempt < self.max_attempts:
                draft_steps = candidate_steps
                draft_final_operation = candidate_operation
                validation_error = None
                continue
            if candidate_steps:
                break
        return ChainGeneration(
            steps=steps,
            final_operation=final_operation,
            llm_calls=len(call_records),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            call_records=tuple(call_records),
            fallback=not steps,
        )


class OpenAIIntermediateAnswerer:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        system_prompt: str,
        evidence_limit: int = 12,
        evidence_char_limit: int = 700,
        same_evidence_retry_limit: int = 0,
    ) -> None:
        prompt = system_prompt.strip()
        if not prompt:
            raise ValueError("intermediate-answer system prompt must be non-empty")
        self.client = client
        self.system_prompt = prompt
        self.system_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.evidence_limit = evidence_limit
        self.evidence_char_limit = evidence_char_limit
        if same_evidence_retry_limit < 0 or same_evidence_retry_limit > 1:
            raise ValueError("same_evidence_retry_limit must be 0 or 1")
        self.same_evidence_retry_limit = same_evidence_retry_limit

    def _evidence_state(self, retrieval_state: RetrievalState) -> list[dict[str, str]]:
        # The old chain answerer was grounded in the current step's top
        # evidence first. Keep that local context ahead of historical evidence
        # so a wrong earlier bridge does not dominate the next answer.
        ordered = list(retrieval_state.current_evidence)
        retained_ids = {
            item.evidence_id for item in retrieval_state.retained_evidence
        }
        ordered.extend(
            candidate.evidence
            for candidate in retrieval_state.candidate_pool
            if candidate.evidence.evidence_id in retained_ids
        )
        ordered.extend(retrieval_state.retained_evidence)
        evidence: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in ordered:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            evidence.append(
                {
                    "title": item.title,
                    "text": item.text[: self.evidence_char_limit],
                }
            )
            if len(evidence) >= self.evidence_limit:
                break
        return evidence

    @staticmethod
    def _parse_answer(raw_output: str) -> str:
        parsed = _json_object(raw_output)
        if parsed is not None:
            return re.sub(r"\s+", " ", str(parsed.get("answer") or "")).strip()
        text = raw_output.strip()
        marker = re.search(r"\bAnswer\s*:\s*", text, flags=re.IGNORECASE)
        if marker is not None:
            recovered = text[marker.end() :].strip()
            recovered = re.sub(r"\s+", " ", recovered).strip(" `\"'")
            return recovered if recovered and len(recovered) <= 240 else ""
        if "\n" not in text and text and len(text) <= 160:
            return text.strip(" `\"'")
        return ""

    def answer(
        self,
        payload: InferencePayload,
        step: GeneratedSignalStep,
        retrieval_state: RetrievalState,
    ) -> IntermediateAnswerGeneration:
        user_prompt = json.dumps(
            {
                "original_question": payload.question,
                "intermediate_question": step.text,
                "retained_evidence": self._evidence_state(retrieval_state),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        call_records: list[dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        answer = ""
        for attempt in range(1, self.same_evidence_retry_limit + 2):
            attempt_prompt = user_prompt
            if attempt > 1:
                attempt_prompt += (
                    "\n\nThe previous response was empty or invalid. Answer the "
                    "same intermediate question from the same evidence. Return "
                    "strict JSON with a non-empty answer string."
                )
            response = self.client.complete(
                (
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=attempt_prompt),
                ),
                temperature=0.0,
                max_completion_tokens=300,
            )
            answer = self._parse_answer(response.content)
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            request_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "system_prompt_sha256": self.system_prompt_sha256,
                        "user_prompt": attempt_prompt,
                    }
                )
            ).hexdigest()
            call_records.append(
                {
                    "kind": "intermediate_answer_generation",
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
                    "step_id": step.step_id,
                    "iteration": retrieval_state.iteration,
                    "format_attempt": attempt,
                    "same_evidence_retry": attempt > 1,
                    "fallback": not answer,
                }
            )
            if answer:
                break
        return IntermediateAnswerGeneration(
            answer=answer,
            llm_calls=len(call_records),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            call_records=tuple(call_records),
            fallback=not answer,
        )


class OpenAITerminalAnswerer(OpenAIIntermediateAnswerer):
    """Answer the terminal chain question using the normal answer prompt.

    This preserves the legacy SC contract: every executed chain step produces
    an answer, and the terminal step's answer is returned directly. The
    answer request still contains only the runtime payload, terminal question,
    and local retrieval evidence.
    """

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        system_prompt: str,
        evidence_limit: int = 12,
        evidence_char_limit: int = 700,
    ) -> None:
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            evidence_limit=evidence_limit,
            evidence_char_limit=evidence_char_limit,
        )
        self.final_generator = OpenAIAnswerGenerator(
            client=client,
            system_prompt=system_prompt,
        )

    def _evidence_units(self, retrieval_state: RetrievalState):
        ordered = list(retrieval_state.current_evidence)
        retained_ids = {
            item.evidence_id for item in retrieval_state.retained_evidence
        }
        ordered.extend(
            candidate.evidence
            for candidate in retrieval_state.candidate_pool
            if candidate.evidence.evidence_id in retained_ids
        )
        ordered.extend(retrieval_state.retained_evidence)
        evidence = []
        seen: set[str] = set()
        for item in ordered:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            evidence.append(item)
            if len(evidence) >= self.evidence_limit:
                break
        return tuple(evidence)

    def answer(
        self,
        payload: InferencePayload,
        step: GeneratedSignalStep,
        retrieval_state: RetrievalState,
    ) -> IntermediateAnswerGeneration:
        request = AnswerRequest(
            payload=payload,
            evidence=self._evidence_units(retrieval_state),
            question_override=step.text,
        )
        try:
            generation = self.final_generator.generate_audited(request)
        except AnswerGenerationError as exc:
            return IntermediateAnswerGeneration(
                answer="",
                llm_calls=exc.llm_calls,
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
                call_records=tuple(exc.call_records),
                fallback=True,
            )
        return IntermediateAnswerGeneration(
            answer=generation.answer,
            llm_calls=generation.llm_calls,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            call_records=generation.call_records,
        )


class SingleStepChainGenerator:
    def generate(
        self,
        payload: InferencePayload,
        max_steps: int,
    ) -> tuple[GeneratedSignalStep, ...]:
        return (GeneratedSignalStep(step_id="step_1", text=payload.question),)


@dataclass(frozen=True, slots=True)
class StaticChainGenerator:
    steps: tuple[GeneratedSignalStep, ...]

    def __init__(self, steps: Sequence[GeneratedSignalStep]) -> None:
        frozen = tuple(steps)
        if not frozen:
            raise ValueError("a static chain must contain at least one step")
        object.__setattr__(self, "steps", frozen)

    def generate(
        self,
        payload: InferencePayload,
        max_steps: int,
    ) -> tuple[GeneratedSignalStep, ...]:
        return self.steps[:max_steps]


class TraceReplayChainGenerator:
    """Replay question-only plans from a prior retrieval trace for controlled A/B runs."""

    def __init__(self, traces: Sequence[Mapping[str, Any]]) -> None:
        chains: dict[str, tuple[GeneratedSignalStep, ...]] = {}
        for row in traces:
            run = row.get("run")
            if not isinstance(run, Mapping):
                continue
            payload = run.get("payload")
            controller_state = run.get("controller_state")
            if not isinstance(payload, Mapping) or not isinstance(
                controller_state,
                Mapping,
            ):
                continue
            question = self._question_key(payload.get("question"))
            raw_chain = controller_state.get("generated_chain")
            if not question or not isinstance(raw_chain, list) or not raw_chain:
                continue
            chain = tuple(
                GeneratedSignalStep.model_validate(step)
                for step in raw_chain
                if isinstance(step, Mapping)
            )
            if len(chain) != len(raw_chain):
                raise ValueError("trace generated_chain contains a non-object step")
            previous = chains.get(question)
            if previous is not None and previous != chain:
                raise ValueError("trace contains conflicting plans for one question")
            chains[question] = chain
        if not chains:
            raise ValueError("trace replay source contains no generated chains")
        self._chains = chains

    @staticmethod
    def _question_key(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

    def generate(
        self,
        payload: InferencePayload,
        max_steps: int,
    ) -> tuple[GeneratedSignalStep, ...]:
        question = self._question_key(payload.question)
        chain = self._chains.get(question)
        if chain is None:
            raise ValueError("no replayed chain exists for the inference question")
        return chain[:max_steps]


class EvidenceTitleAnswerer:
    """Offline answer hook used when no intermediate LLM is configured."""

    def answer(
        self,
        payload: InferencePayload,
        step: GeneratedSignalStep,
        retrieval_state: RetrievalState,
    ) -> str:
        if not retrieval_state.candidate_pool:
            return ""
        return retrieval_state.candidate_pool[0].evidence.title.strip()


@dataclass(frozen=True, slots=True)
class StaticIntermediateAnswerer:
    answers: Mapping[str, str]

    def answer(
        self,
        payload: InferencePayload,
        step: GeneratedSignalStep,
        retrieval_state: RetrievalState,
    ) -> str:
        return str(self.answers.get(step.step_id, "")).strip()


def materialize_step(
    step: GeneratedSignalStep,
    *,
    chain: tuple[GeneratedSignalStep, ...],
    answers: Mapping[str, str],
) -> tuple[str, tuple[str, ...], bool]:
    text = step.text
    used_answers: list[str] = []
    ordinal_to_id = {index: item.step_id for index, item in enumerate(chain, start=1)}

    def replace_numbered(match: re.Match[str]) -> str:
        step_id = ordinal_to_id.get(int(match.group(1)))
        answer = answers.get(step_id or "", "").strip()
        if answer:
            used_answers.append(answer)
            return answer
        return match.group(0)

    text = _NUMBERED_PLACEHOLDER.sub(replace_numbered, text)
    for dependency in step.depends_on:
        answer = answers.get(dependency, "").strip()
        if not answer:
            continue
        placeholder = f"<answer_{dependency}>"
        if placeholder in text:
            text = text.replace(placeholder, answer)
            used_answers.append(answer)

    if not used_answers:
        dependency_answers = [
            answers[dependency].strip()
            for dependency in step.depends_on
            if answers.get(dependency, "").strip()
        ]
        if dependency_answers:
            answer = dependency_answers[-1]
            replaced = re.sub(
                r"\b(?:this|that|the)\s+(?:person|entity|individual|place|location|city|country|organization)\b",
                answer,
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            if replaced != text:
                text = replaced
                used_answers.append(answer)

    text = re.sub(r"\s+", " ", text).strip()
    unresolved = bool(_NUMBERED_PLACEHOLDER.search(text) or re.search(r"<answer_[^>]+>", text))
    return text, tuple(dict.fromkeys(used_answers)), unresolved


class SubquestionChainController(SignalController):
    def __init__(
        self,
        *,
        generator: ChainGenerator | None = None,
        answerer: IntermediateAnswerer | None = None,
        final_answerer: IntermediateAnswerer | None = None,
        max_chain_steps: int | None = None,
        max_step_retries: int = 0,
    ) -> None:
        if max_chain_steps is not None and max_chain_steps < 1:
            raise ValueError("max_chain_steps must be positive when provided")
        if max_step_retries < 0:
            raise ValueError("max_step_retries must be non-negative")
        self.generator = generator or SingleStepChainGenerator()
        self.answerer = answerer or EvidenceTitleAnswerer()
        self.final_answerer = final_answerer
        self.max_chain_steps = max_chain_steps
        self.max_step_retries = max_step_retries
        self._generation = ChainGeneration(steps=())
        self._logical_step_index = 0
        self._retry_pending = False
        self._step_retry_counts: dict[int, int] = {}

    @property
    def controller_type(self) -> str:
        return "subquestion_chain"

    def _start(self, payload: InferencePayload, budget: BudgetContract) -> ControllerState:
        self._logical_step_index = 0
        self._retry_pending = False
        self._step_retry_counts = {}
        max_chain_steps = min(
            budget.max_iterations,
            self.max_chain_steps or budget.max_iterations,
        )
        generated = self.generator.generate(payload, max_chain_steps)
        if isinstance(generated, ChainGeneration):
            self._generation = generated
            chain = generated.steps
        else:
            self._generation = ChainGeneration(steps=tuple(generated))
            chain = self._generation.steps
        if not chain:
            chain = SingleStepChainGenerator().generate(payload, budget.max_iterations)
        return ControllerState.started(
            controller_type=self.controller_type,
            budget=budget,
            generated_chain=chain,
            final_operation=self._generation.final_operation,
        )

    @staticmethod
    def _blocked_descendants(
        step_id: str,
        chain: tuple[GeneratedSignalStep, ...],
    ) -> tuple[str, ...]:
        blocked: set[str] = set()
        changed = True
        while changed:
            changed = False
            for candidate in chain:
                if candidate.step_id in blocked:
                    continue
                if step_id in candidate.depends_on or any(
                    dependency in blocked for dependency in candidate.depends_on
                ):
                    blocked.add(candidate.step_id)
                    changed = True
        return tuple(
            step.step_id
            for step in chain
            if step.step_id in blocked and step.step_id != step_id
        )

    def _has_unblocked_step(
        self,
        state: ControllerState,
        start_index: int,
        additional_blocked: Sequence[str] = (),
    ) -> bool:
        blocked = set(state.blocked_step_ids) | set(additional_blocked)
        return any(
            step.step_id not in blocked
            for step in state.generated_chain[start_index:]
        )

    @staticmethod
    def _fallback_signal_text(
        payload: InferencePayload,
        step: GeneratedSignalStep,
        materialized: str,
        answers: Mapping[str, str],
        retrieval_state: RetrievalState,
    ) -> str:
        """Build the old chain-break context without leaking annotations.

        The fallback keeps the current relation as the retrieval focus, while
        giving the retriever the original question and the small amount of
        dependency/title context that was available to the legacy runner.
        """
        relation = _NUMBERED_PLACEHOLDER.sub("the unresolved entity", materialized)
        relation = re.sub(r"<answer_[^>]+>", "the unresolved entity", relation)
        relation = re.sub(r"\s+", " ", relation).strip() or payload.question
        dependency_answers = [
            answers[dependency].strip()
            for dependency in step.depends_on
            if answers.get(dependency, "").strip()
        ]
        titles = list(
            dict.fromkeys(
                evidence.title.strip()
                for evidence in retrieval_state.current_evidence
                if evidence.title.strip()
            )
        )[:10]
        context = [f"Current relation: {relation}", f"Original question: {payload.question}"]
        if dependency_answers:
            context.append("Dependency answers: " + "; ".join(dependency_answers))
        if titles:
            context.append("Previous evidence titles: " + "; ".join(titles))
        return "\n".join(context)

    def _produce_signal(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
    ) -> SignalDecision:
        blocked = set(state.blocked_step_ids)
        while (
            self._logical_step_index < len(state.generated_chain)
            and state.generated_chain[self._logical_step_index].step_id in blocked
        ):
            self._logical_step_index += 1
        if self._logical_step_index >= len(state.generated_chain):
            reason = "all remaining chain steps are blocked by failed dependencies"
            return SignalDecision(
                decision_id=decision_id(
                    self.controller_type,
                    state.completed_steps + 1,
                    reason,
                ),
                controller_type=self.controller_type,
                iteration=state.completed_steps + 1,
                action="stop",
                reason=reason,
                selected_by=type(self.generator).__name__,
            )
        step = state.generated_chain[self._logical_step_index]
        answers = {answer.step_id: answer.answer for answer in state.step_answers}
        materialized, _, unresolved = materialize_step(
            step,
            chain=state.generated_chain,
            answers=answers,
        )
        generation = self._generation if state.completed_steps == 0 else ChainGeneration(steps=())
        fallback = self._retry_pending or unresolved or not materialized or generation.fallback
        if self._retry_pending or unresolved:
            signal_text = self._fallback_signal_text(
                payload,
                step,
                materialized,
                answers,
                retrieval_state,
            )
        else:
            signal_text = payload.question if fallback else materialized
        iteration = state.completed_steps + 1
        return SignalDecision(
            decision_id=decision_id(self.controller_type, iteration, signal_text),
            controller_type=self.controller_type,
            iteration=iteration,
            action="retrieve",
            reason=(
                "retry current chain step after an empty intermediate answer"
                if self._retry_pending
                else "fallback to the current relation with original-question context"
                if fallback
                else f"execute generated chain step {step.step_id}"
            ),
            signal_text=signal_text,
            selected_by=type(self.generator).__name__,
            selected_step_id=step.step_id,
            fallback=fallback,
            llm_calls=generation.llm_calls,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            call_records=tuple(dict(record) for record in generation.call_records),
        )

    def _consume_evidence(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
        decision: SignalDecision,
    ) -> StepOutcome:
        step_index = self._logical_step_index
        step = state.generated_chain[step_index]
        final_step = step_index + 1 >= len(state.generated_chain)
        terminal_answer = (
            final_step
            and self.final_answerer is not None
            and state.final_operation is None
        )
        needs_answer = terminal_answer or any(
            step.step_id in later.depends_on
            for later in state.generated_chain[step_index + 1 :]
        )
        was_retry = self._retry_pending
        answer = ""
        llm_calls = 0
        input_tokens = 0
        output_tokens = 0
        call_records: tuple[Mapping[str, Any], ...] = ()
        if needs_answer:
            materialized_step = step.model_copy(update={"text": decision.signal_text})
            answerer = self.final_answerer if terminal_answer else self.answerer
            generated_answer = answerer.answer(
                payload,
                materialized_step,
                retrieval_state,
            )
            if isinstance(generated_answer, IntermediateAnswerGeneration):
                answer = generated_answer.answer.strip()
                llm_calls = generated_answer.llm_calls
                input_tokens = generated_answer.input_tokens
                output_tokens = generated_answer.output_tokens
                call_records = generated_answer.call_records
            else:
                answer = generated_answer.strip()
        retained_ids = {
            evidence.evidence_id for evidence in retrieval_state.retained_evidence
        }
        consumed = tuple(
            candidate.evidence.evidence_id
            for candidate in retrieval_state.candidate_pool
            if candidate.evidence.evidence_id in retained_ids
        )
        schedule_retry = False
        failed_step_id: str | None = None
        blocked_step_ids: tuple[str, ...] = ()
        step_status = "succeeded"
        next_action = "stop" if final_step else "continue"
        if needs_answer and not answer:
            retries_used = self._step_retry_counts.get(step_index, 0)
            completed_attempts = state.completed_steps + 1
            remaining_attempt_budget = state.budget.max_iterations - completed_attempts
            attempts_needed_after_retry = len(state.generated_chain) - step_index
            schedule_retry = (
                not was_retry
                and retries_used < self.max_step_retries
                and remaining_attempt_budget >= attempts_needed_after_retry
            )
        if schedule_retry:
            self._step_retry_counts[step_index] = (
                self._step_retry_counts.get(step_index, 0) + 1
            )
            self._retry_pending = True
            step_status = "retrying"
        else:
            self._retry_pending = False
            self._logical_step_index += 1
            if needs_answer and not answer:
                failed_step_id = step.step_id
                blocked_step_ids = self._blocked_descendants(
                    step.step_id,
                    state.generated_chain,
                )
                step_status = "failed"
                next_action = (
                    "continue"
                    if self._has_unblocked_step(
                        state,
                        self._logical_step_index,
                        blocked_step_ids,
                    )
                    else "stop"
                )
            else:
                next_action = (
                    "continue"
                    if self._has_unblocked_step(state, self._logical_step_index)
                    else "stop"
                )

        return StepOutcome.create(
            state=state,
            decision=decision,
            consumed_evidence_ids=consumed,
            intermediate_answer=answer or None,
            answer_step_id=step.step_id if answer else None,
            step_status=step_status,
            failed_step_id=failed_step_id,
            blocked_step_ids=blocked_step_ids,
            next_action=next_action,
            reason=(
                "generated chain exhausted"
                if next_action == "stop" and not failed_step_id
                else "empty intermediate answer; retry current chain step"
                if schedule_retry
                else "advance generated chain"
                if answer
                else "advance after independent terminal retrieval"
                if not needs_answer
                else "chain-step failed; dependent steps are blocked"
            ),
            llm_calls=llm_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_records=call_records,
        )


__all__ = [
    "ChainGenerator",
    "ChainGeneration",
    "EvidenceTitleAnswerer",
    "IntermediateAnswerer",
    "IntermediateAnswerGeneration",
    "OpenAIChainGenerator",
    "OpenAIIntermediateAnswerer",
    "OpenAITerminalAnswerer",
    "SingleStepChainGenerator",
    "StaticChainGenerator",
    "StaticIntermediateAnswerer",
    "SubquestionChainController",
    "TraceReplayChainGenerator",
    "materialize_step",
]
