from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from eager.answering import (
    AnswerGeneration,
    AnswerGenerationError,
    AnswerGenerator,
    AnswerRequest,
)
from eager.config import canonical_json_bytes
from eager.data import EvaluationBatch, InferenceBatch
from eager.evaluation import (
    AnswerMetrics,
    JudgeEvaluation,
    SupportMetrics,
    evaluate_answer,
    evaluate_support,
)
from eager.pipeline import OnlineAutoRunner, RetrievalEngine, RetrievalRunResult
from eager.protocol import ExperimentProtocol, protocol_sha256
from eager.runtime.checkpoint import CheckpointMismatchError, CheckpointStore, ProvenanceBinding
from eager.runtime.io import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_json,
)
from eager.runtime.logging import redact
from eager.runtime.manifest import environment_snapshot, git_snapshot, output_inventory, write_manifest
from eager.schemas import (
    BudgetContract,
    BudgetUsage,
    EvaluationAnnotations,
    EvidenceUnit,
    InferencePayload,
    PredictionError,
    PredictionRecord,
)
from eager.signals import SignalController


class ExperimentRunError(RuntimeError):
    pass


class RetrievalExecutor(Protocol):
    def run(
        self,
        *,
        payload: InferencePayload,
        budget: BudgetContract,
    ) -> RetrievalRunResult: ...


class AnswerJudge(Protocol):
    """Evaluator-only judge. Implementations must not enter the inference path."""

    def judge(
        self,
        *,
        prediction: str,
        annotations: EvaluationAnnotations,
    ) -> bool: ...


class FixedPolicyExecutor:
    def __init__(
        self,
        engine: RetrievalEngine,
        controller_factory: Callable[[], SignalController],
    ) -> None:
        self.engine = engine
        self.controller_factory = controller_factory

    def run(
        self,
        *,
        payload: InferencePayload,
        budget: BudgetContract,
    ) -> RetrievalRunResult:
        return self.engine.run(
            payload=payload,
            controller=self.controller_factory(),
            budget=budget,
        )


class AutoPolicyExecutor:
    def __init__(self, runner: OnlineAutoRunner) -> None:
        self.runner = runner

    def run(
        self,
        *,
        payload: InferencePayload,
        budget: BudgetContract,
    ) -> RetrievalRunResult:
        return self.runner.run(payload=payload, budget=budget).run


class QuestionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    handle_id: str
    answer: AnswerMetrics
    support: SupportMetrics
    candidate_support: SupportMetrics
    support_at: dict[str, SupportMetrics]
    last_gold_support_rank: dict[str, int | None]
    gpt_accuracy: bool | None = None
    retrieval_latency_ms: float = Field(ge=0.0)
    end_to_end_latency_ms: float = Field(ge=0.0)
    candidate_count: int = Field(ge=0)
    parent_unit_count: int = Field(ge=0)
    rerank_pairs: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    planner_call_count: int = Field(default=0, ge=0)
    planner_retry_count: int = Field(default=0, ge=0)
    planner_rejected_count: int = Field(default=0, ge=0)
    intermediate_answer_retry_count: int = Field(default=0, ge=0)
    failed_step_count: int = Field(default=0, ge=0)
    blocked_step_count: int = Field(default=0, ge=0)
    status: str = "ok"
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    prediction_count: int
    resumed_count: int
    status: str
    metrics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class QuestionExecution:
    prediction: PredictionRecord
    trace: dict[str, Any]
    metrics: QuestionMetrics
    call_records: tuple[dict[str, Any], ...]


def validate_protocol_budget(
    protocol: ExperimentProtocol,
    *,
    dataset: str,
    profile: str,
    budget: BudgetContract,
) -> None:
    if dataset not in protocol.datasets:
        raise ExperimentRunError(f"dataset is not registered by the protocol: {dataset}")
    if profile not in protocol.execution_profiles:
        raise ExperimentRunError(f"execution profile is not registered: {profile}")
    if budget.max_candidates_per_step not in protocol.evaluation.candidate_budgets:
        raise ExperimentRunError(
            "candidate step budget is not one of the protocol candidate_budgets"
        )
    if budget.max_retained_evidence not in protocol.evaluation.final_evidence_budgets:
        raise ExperimentRunError(
            "retained evidence budget is not one of the protocol final_evidence_budgets"
        )


def _mean(rows: Sequence[QuestionMetrics], field: str) -> float:
    return sum(float(getattr(row, field)) for row in rows) / len(rows) if rows else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _optional_bool_rate(values: Sequence[bool | None]) -> dict[str, Any]:
    applicable = [value for value in values if value is not None]
    return {
        "value": (
            sum(bool(value) for value in applicable) / len(applicable)
            if applicable
            else None
        ),
        "applicable_question_count": len(applicable),
    }


def _last_gold_rank(
    evidence: Sequence[EvidenceUnit],
    gold_ids: Sequence[str],
    *,
    field: str,
) -> int | None:
    if not gold_ids:
        return None
    ranks: dict[str, int] = {}
    for rank, item in enumerate(evidence, start=1):
        value = str(getattr(item, field))
        ranks.setdefault(value, rank)
    if any(gold_id not in ranks for gold_id in gold_ids):
        return None
    return max(ranks[gold_id] for gold_id in gold_ids)


def _candidate_evidence(retrieved: RetrievalRunResult) -> tuple[EvidenceUnit, ...]:
    output: list[EvidenceUnit] = []
    seen: set[str] = set()
    for step in retrieved.steps:
        candidates = step.retrieval.coarse_candidates or step.retrieval.candidates
        for candidate in candidates:
            evidence = candidate.evidence
            if evidence.evidence_id not in seen:
                seen.add(evidence.evidence_id)
                output.append(evidence)
    return tuple(output)


def _controller_chain_metrics(
    retrieved: RetrievalRunResult | None,
) -> dict[str, int]:
    """Summarize planner and step-state transitions from the immutable trace."""
    counts = {
        "planner_call_count": 0,
        "planner_retry_count": 0,
        "planner_rejected_count": 0,
        "intermediate_answer_retry_count": 0,
        "failed_step_count": 0,
        "blocked_step_count": 0,
    }
    if retrieved is None:
        return counts
    for trace_step in retrieved.steps:
        for record in trace_step.decision.call_records:
            if record.get("kind") != "subquestion_decomposition":
                continue
            counts["planner_call_count"] += 1
            if int(record.get("plan_attempt") or 1) > 1:
                counts["planner_retry_count"] += 1
            if record.get("validation_error"):
                counts["planner_rejected_count"] += 1
        status = trace_step.outcome.step_status
        if status == "retrying":
            counts["intermediate_answer_retry_count"] += 1
        elif status == "failed":
            counts["failed_step_count"] += 1
    # A blocked step is carried in every later state, so count unique ids.
    counts["blocked_step_count"] = len(retrieved.controller_state.blocked_step_ids)
    return counts


def _aggregate(rows: Sequence[QuestionMetrics]) -> dict[str, Any]:
    count = len(rows)
    if not rows:
        return {"question_count": 0}
    candidate_counts = [float(row.candidate_count) for row in rows]
    retrieval_latencies = [row.retrieval_latency_ms for row in rows]
    end_to_end_latencies = [row.end_to_end_latency_ms for row in rows]
    gpt_accuracy = _optional_bool_rate([row.gpt_accuracy for row in rows])
    sentence_candidate_any = _optional_bool_rate(
        [row.candidate_support.sentence.any_recall for row in rows]
    )
    sentence_candidate_full = _optional_bool_rate(
        [row.candidate_support.sentence.full_recall for row in rows]
    )
    paragraph_candidate_any = _optional_bool_rate(
        [row.candidate_support.paragraph.any_recall for row in rows]
    )
    paragraph_candidate_full = _optional_bool_rate(
        [row.candidate_support.paragraph.full_recall for row in rows]
    )
    full_support_at: dict[str, dict[str, dict[str, Any]]] = {
        "sentence": {},
        "paragraph": {},
    }
    for limit in (5, 10, 20):
        key = str(limit)
        full_support_at["sentence"][key] = _optional_bool_rate(
            [row.support_at[key].sentence.full_recall for row in rows]
        )
        full_support_at["paragraph"][key] = _optional_bool_rate(
            [row.support_at[key].paragraph.full_recall for row in rows]
        )
    sentence_ranks = [
        float(rank)
        for row in rows
        if (rank := row.last_gold_support_rank["sentence"]) is not None
    ]
    paragraph_ranks = [
        float(rank)
        for row in rows
        if (rank := row.last_gold_support_rank["paragraph"]) is not None
    ]
    error_codes: dict[str, int] = {}
    for row in rows:
        if row.error_code is not None:
            error_codes[row.error_code] = error_codes.get(row.error_code, 0) + 1
    error_count = sum(row.status == "error" for row in rows)
    return {
        "question_count": count,
        "successful_question_count": count - error_count,
        "error_question_count": error_count,
        "error_rate": error_count / count,
        "error_codes": error_codes,
        "normalized_em": sum(row.answer.exact_match for row in rows) / count,
        "answer_f1": sum(row.answer.f1 for row in rows) / count,
        "containment_accuracy": sum(
            row.answer.containment_accuracy for row in rows
        )
        / count,
        "sentence_any_support_recall": _optional_bool_rate(
            [row.support.sentence.any_recall for row in rows]
        ),
        "sentence_full_support_recall": _optional_bool_rate(
            [row.support.sentence.full_recall for row in rows]
        ),
        "paragraph_any_support_recall": _optional_bool_rate(
            [row.support.paragraph.any_recall for row in rows]
        ),
        "paragraph_full_support_recall": _optional_bool_rate(
            [row.support.paragraph.full_recall for row in rows]
        ),
        "candidate_any_support_recall": {
            "sentence": sentence_candidate_any,
            "paragraph": paragraph_candidate_any,
        },
        "candidate_full_support_recall": {
            "sentence": sentence_candidate_full,
            "paragraph": paragraph_candidate_full,
        },
        "full_support_at_5_10_20": full_support_at,
        "last_gold_support_rank": {
            "sentence": {
                **_distribution(sentence_ranks),
                "applicable_question_count": len(sentence_ranks),
            },
            "paragraph": {
                **_distribution(paragraph_ranks),
                "applicable_question_count": len(paragraph_ranks),
            },
        },
        "candidate_count_mean": _mean(rows, "candidate_count"),
        "candidate_count_mean_p50_p95": _distribution(candidate_counts),
        "parent_unit_count": sum(row.parent_unit_count for row in rows),
        "reranker_pair_count": sum(row.rerank_pairs for row in rows),
        "retrieval_latency_mean_ms": _mean(rows, "retrieval_latency_ms"),
        "retrieval_latency_p50_p95": _distribution(retrieval_latencies),
        "end_to_end_latency_mean_ms": _mean(rows, "end_to_end_latency_ms"),
        "end_to_end_latency_p50_p95": _distribution(end_to_end_latencies),
        "llm_calls": sum(row.llm_calls for row in rows),
        "input_tokens": sum(row.input_tokens for row in rows),
        "output_tokens": sum(row.output_tokens for row in rows),
        "llm_calls_and_tokens": {
            "calls": sum(row.llm_calls for row in rows),
            "input_tokens": sum(row.input_tokens for row in rows),
            "output_tokens": sum(row.output_tokens for row in rows),
        },
        "controller_chain": {
            "planner_call_count": sum(row.planner_call_count for row in rows),
            "planner_retry_count": sum(row.planner_retry_count for row in rows),
            "planner_rejected_count": sum(row.planner_rejected_count for row in rows),
            "intermediate_answer_retry_count": sum(
                row.intermediate_answer_retry_count for row in rows
            ),
            "failed_step_count": sum(row.failed_step_count for row in rows),
            "blocked_step_count": sum(row.blocked_step_count for row in rows),
        },
        "gpt_accuracy": gpt_accuracy["value"],
        "gpt_accuracy_status": (
            "complete" if gpt_accuracy["applicable_question_count"] == count else "judge_not_configured"
        ),
        "gpt_accuracy_question_count": gpt_accuracy["applicable_question_count"],
    }


def _report(metrics: Mapping[str, Any]) -> str:
    lines = ["# Experiment report", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


class ExperimentRunner:
    def __init__(
        self,
        *,
        retrieval: RetrievalExecutor,
        answer_generator: AnswerGenerator,
        protocol: ExperimentProtocol,
        repo_root: Path,
        answer_judge: AnswerJudge | None = None,
        final_answer_scope: str = "global",
        git_snapshotter: Callable[[Path], Mapping[str, Any]] = git_snapshot,
        environment_snapshotter: Callable[[], Mapping[str, Any]] = environment_snapshot,
    ) -> None:
        self.retrieval = retrieval
        self.answer_generator = answer_generator
        self.protocol = protocol
        self.repo_root = repo_root
        self.answer_judge = answer_judge
        if final_answer_scope not in {"global", "last_subquestion"}:
            raise ValueError(
                "final_answer_scope must be 'global' or 'last_subquestion'"
            )
        self.final_answer_scope = final_answer_scope
        self.git_snapshotter = git_snapshotter
        self.environment_snapshotter = environment_snapshotter

    def _failed_question(
        self,
        *,
        handle_id: str,
        annotations: EvaluationAnnotations,
        budget: BudgetContract,
        error: Exception,
        question_started: float,
        retrieval_latency_ms: float = 0.0,
        retrieved: RetrievalRunResult | None = None,
        answer_usage: BudgetUsage | None = None,
        call_records: Sequence[Mapping[str, Any]] = (),
    ) -> QuestionExecution:
        safe_message = redact(str(error))
        if not isinstance(safe_message, str) or not safe_message.strip():
            safe_message = "question execution failed"
        raw_code = getattr(error, "code", None)
        if isinstance(error, AnswerGenerationError):
            error_code = "answer_format_error"
        elif isinstance(raw_code, str) and raw_code.strip():
            error_code = raw_code.strip()
        else:
            error_code = "question_execution_error"
        evidence = (
            retrieved.retrieval_state.retained_evidence if retrieved is not None else ()
        )
        signal_trace = (
            retrieved.retrieval_state.signal_history if retrieved is not None else ()
        )
        step_trace = (
            tuple(step.outcome for step in retrieved.steps) if retrieved is not None else ()
        )
        chain_metrics = _controller_chain_metrics(retrieved)
        candidate_evidence = _candidate_evidence(retrieved) if retrieved is not None else ()
        retrieval_usage = (
            retrieved.retrieval_state.total_usage if retrieved is not None else BudgetUsage()
        )
        failed_answer_usage = answer_usage or BudgetUsage()
        total_usage = BudgetUsage(
            iterations=retrieval_usage.iterations,
            candidates_considered=retrieval_usage.candidates_considered,
            parent_units_selected=retrieval_usage.parent_units_selected,
            rerank_pairs=retrieval_usage.rerank_pairs,
            llm_calls=retrieval_usage.llm_calls + failed_answer_usage.llm_calls,
            input_tokens=retrieval_usage.input_tokens + failed_answer_usage.input_tokens,
            output_tokens=retrieval_usage.output_tokens + failed_answer_usage.output_tokens,
        )
        end_to_end_latency_ms = (time.perf_counter() - question_started) * 1000.0
        prediction_error = PredictionError(
            code=error_code,
            message=safe_message,
            retryable=bool(getattr(error, "retryable", False)),
        )
        prediction = PredictionRecord(
            handle_id=handle_id,
            controller_type=(retrieved.controller_type if retrieved is not None else "question_error"),
            status="error",
            evidence=evidence,
            signal_trace=signal_trace,
            step_trace=step_trace,
            budget=budget,
            usage=total_usage,
            latency_ms=end_to_end_latency_ms,
            error=prediction_error,
        )
        support = evaluate_support(list(evidence), annotations, limit=budget.max_retained_evidence)
        candidate_support = evaluate_support(list(candidate_evidence), annotations)
        metrics = QuestionMetrics(
            handle_id=handle_id,
            answer=evaluate_answer("", annotations.answer, aliases=annotations.answer_aliases),
            support=support,
            candidate_support=candidate_support,
            support_at={
                str(limit): evaluate_support(list(evidence), annotations, limit=limit)
                for limit in (5, 10, 20)
            },
            last_gold_support_rank={
                "sentence": _last_gold_rank(
                    candidate_evidence,
                    annotations.supporting_sentence_ids,
                    field="sentence_id",
                ),
                "paragraph": _last_gold_rank(
                    candidate_evidence,
                    annotations.supporting_paragraph_ids,
                    field="paragraph_id",
                ),
            },
            gpt_accuracy=False if self.answer_judge is not None else None,
            retrieval_latency_ms=retrieval_latency_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            candidate_count=total_usage.candidates_considered,
            parent_unit_count=total_usage.parent_units_selected,
            rerank_pairs=total_usage.rerank_pairs,
            llm_calls=total_usage.llm_calls,
            input_tokens=total_usage.input_tokens,
            output_tokens=total_usage.output_tokens,
            **chain_metrics,
            status="error",
            error_code=error_code,
        )
        safe_calls: list[dict[str, Any]] = []
        for call_record in call_records:
            safe_call = redact({"handle_id": handle_id, **dict(call_record)})
            if isinstance(safe_call, dict):
                safe_calls.append(safe_call)
        return QuestionExecution(
            prediction=prediction,
            trace={
                "handle_id": handle_id,
                "run": retrieved.model_dump(mode="json") if retrieved is not None else None,
                "error": prediction_error.model_dump(mode="json"),
            },
            metrics=metrics,
            call_records=tuple(safe_calls),
        )

    def _process_question(
        self,
        *,
        handle_id: str,
        payload: InferencePayload,
        annotations: EvaluationAnnotations,
        budget: BudgetContract,
    ) -> QuestionExecution:
        question_started = time.perf_counter()
        question_calls: list[dict[str, Any]] = []
        retrieval_started = time.perf_counter()
        retrieved = self.retrieval.run(payload=payload, budget=budget)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
        for trace_step in retrieved.steps:
            for stage, records in (
                ("signal", trace_step.decision.call_records),
                ("outcome", trace_step.outcome.call_records),
            ):
                for record in records:
                    safe_call = redact(
                        {
                            "handle_id": handle_id,
                            "controller_type": retrieved.controller_type,
                            "iteration": trace_step.decision.iteration,
                            "retrieval_stage": stage,
                            **dict(record),
                        }
                    )
                    if not isinstance(safe_call, dict):
                        raise ExperimentRunError("signal LLM call record must be an object")
                    question_calls.append(safe_call)
        try:
            direct_answer = self._direct_terminal_chain_answer(retrieved)
            if direct_answer is not None:
                # The terminal chain answer was already generated inside the
                # controller. Do not invoke a second final-answer LLM.
                generation = AnswerGeneration(answer=direct_answer)
            else:
                answer_request = self._build_answer_request(payload, retrieved)
                audited_generate = getattr(self.answer_generator, "generate_audited", None)
                if callable(audited_generate):
                    generation = audited_generate(answer_request)
                    if not isinstance(generation, AnswerGeneration):
                        raise ExperimentRunError("generate_audited must return AnswerGeneration")
                else:
                    generation = AnswerGeneration(answer=self.answer_generator.generate(answer_request))
        except Exception as exc:
            answer_usage = BudgetUsage(
                llm_calls=int(getattr(exc, "llm_calls", 0)),
                input_tokens=int(getattr(exc, "input_tokens", 0)),
                output_tokens=int(getattr(exc, "output_tokens", 0)),
            )
            return self._failed_question(
                handle_id=handle_id,
                annotations=annotations,
                budget=budget,
                error=exc,
                question_started=question_started,
                retrieval_latency_ms=retrieval_latency_ms,
                retrieved=retrieved,
                answer_usage=answer_usage,
                call_records=(*question_calls, *getattr(exc, "call_records", ())),
            )
        answer = generation.answer.strip()
        if not answer:
            raise ExperimentRunError("answer generator returned an empty answer")
        generation_records = generation.call_records
        if not generation_records and generation.call_record is not None:
            generation_records = (generation.call_record,)
        for call_record in generation_records:
            safe_call = redact({"handle_id": handle_id, **dict(call_record)})
            if not isinstance(safe_call, dict):
                raise ExperimentRunError("LLM call record must be an object")
            question_calls.append(safe_call)
        answer_metrics = evaluate_answer(
            answer,
            annotations.answer,
            aliases=annotations.answer_aliases,
        )
        support_metrics = evaluate_support(
            list(retrieved.retrieval_state.retained_evidence),
            annotations,
            limit=budget.max_retained_evidence,
        )
        candidate_evidence = _candidate_evidence(retrieved)
        candidate_support = evaluate_support(candidate_evidence, annotations)
        support_at = {
            str(limit): evaluate_support(
                list(retrieved.retrieval_state.retained_evidence),
                annotations,
                limit=limit,
            )
            for limit in (5, 10, 20)
        }
        judge_evaluation: JudgeEvaluation | None = None
        if self.answer_judge is None:
            gpt_accuracy = None
        else:
            audited_judge = getattr(self.answer_judge, "judge_audited", None)
            if callable(audited_judge):
                judge_evaluation = audited_judge(
                    prediction=answer,
                    annotations=annotations,
                )
                if not isinstance(judge_evaluation, JudgeEvaluation):
                    raise ExperimentRunError("judge_audited must return JudgeEvaluation")
                gpt_accuracy = judge_evaluation.correct
                safe_call = redact(
                    {
                        "handle_id": handle_id,
                        **dict(judge_evaluation.call_record),
                    }
                )
                if not isinstance(safe_call, dict):
                    raise ExperimentRunError("judge call record must be an object")
                question_calls.append(safe_call)
            else:
                gpt_accuracy = self.answer_judge.judge(
                    prediction=answer,
                    annotations=annotations,
                )
        end_to_end_latency_ms = (time.perf_counter() - question_started) * 1000.0
        retrieval_usage = retrieved.retrieval_state.total_usage
        chain_metrics = _controller_chain_metrics(retrieved)
        total_usage = BudgetUsage(
            iterations=retrieval_usage.iterations,
            candidates_considered=retrieval_usage.candidates_considered,
            parent_units_selected=retrieval_usage.parent_units_selected,
            rerank_pairs=retrieval_usage.rerank_pairs,
            llm_calls=(
                retrieval_usage.llm_calls
                + generation.llm_calls
                + (judge_evaluation.llm_calls if judge_evaluation is not None else 0)
            ),
            input_tokens=(
                retrieval_usage.input_tokens
                + generation.input_tokens
                + (judge_evaluation.input_tokens if judge_evaluation is not None else 0)
            ),
            output_tokens=(
                retrieval_usage.output_tokens
                + generation.output_tokens
                + (judge_evaluation.output_tokens if judge_evaluation is not None else 0)
            ),
        )
        prediction = PredictionRecord(
            handle_id=handle_id,
            controller_type=retrieved.controller_type,
            status="ok",
            answer=answer,
            evidence=retrieved.retrieval_state.retained_evidence,
            signal_trace=retrieved.retrieval_state.signal_history,
            step_trace=tuple(step.outcome for step in retrieved.steps),
            budget=budget,
            usage=total_usage,
            latency_ms=end_to_end_latency_ms,
        )
        trace = {
            "handle_id": handle_id,
            "run": retrieved.model_dump(mode="json"),
        }
        metrics = QuestionMetrics(
            handle_id=handle_id,
            answer=answer_metrics,
            support=support_metrics,
            candidate_support=candidate_support,
            support_at=support_at,
            last_gold_support_rank={
                "sentence": _last_gold_rank(
                    candidate_evidence,
                    annotations.supporting_sentence_ids,
                    field="sentence_id",
                ),
                "paragraph": _last_gold_rank(
                    candidate_evidence,
                    annotations.supporting_paragraph_ids,
                    field="paragraph_id",
                ),
            },
            gpt_accuracy=gpt_accuracy,
            retrieval_latency_ms=retrieval_latency_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            candidate_count=total_usage.candidates_considered,
            parent_unit_count=total_usage.parent_units_selected,
            rerank_pairs=total_usage.rerank_pairs,
            llm_calls=total_usage.llm_calls,
            input_tokens=total_usage.input_tokens,
            output_tokens=total_usage.output_tokens,
            **chain_metrics,
        )
        return QuestionExecution(
            prediction=prediction,
            trace=trace,
            metrics=metrics,
            call_records=tuple(question_calls),
        )

    def _build_answer_request(
        self,
        payload: InferencePayload,
        retrieved: RetrievalRunResult,
    ) -> AnswerRequest:
        """Construct the final answer context without mixing retrieval scopes.

        The legacy SC answer layer answered the terminal materialized
        subquestion from that step's local evidence. Keep this path explicit
        and limited to linear chains; DAG plans retain their operation-aware
        global answer context.
        """
        if (
            self.final_answer_scope == "last_subquestion"
            and retrieved.controller_type == "subquestion_chain"
            and retrieved.controller_state.final_operation is None
            and retrieved.steps
        ):
            terminal = retrieved.steps[-1]
            local_evidence = (
                terminal.retrieval.context_evidence
                or terminal.retrieval.evidence
            )
            question = terminal.decision.signal_text.strip() or payload.question
            return AnswerRequest(
                payload=payload,
                evidence=local_evidence,
                final_operation=None,
                step_answers=(),
                question_override=question,
            )
        return AnswerRequest(
            payload=payload,
            evidence=retrieved.retrieval_state.retained_evidence,
            final_operation=retrieved.controller_state.final_operation,
            step_answers=retrieved.controller_state.step_answers,
        )

    def _direct_terminal_chain_answer(
        self,
        retrieved: RetrievalRunResult,
    ) -> str | None:
        """Return a terminal linear-chain answer already produced in retrieval."""
        if (
            self.final_answer_scope != "last_subquestion"
            or retrieved.controller_type != "subquestion_chain"
            or retrieved.controller_state.final_operation is not None
            or not retrieved.steps
        ):
            return None
        terminal = retrieved.steps[-1].outcome
        if terminal.answer_step_id is None:
            raise AnswerGenerationError(
                "terminal chain step did not produce a non-empty answer",
            )
        answer = (terminal.intermediate_answer or "").strip()
        if not answer:
            raise AnswerGenerationError(
                "terminal chain step did not produce a non-empty answer",
            )
        return answer

    def run(
        self,
        *,
        run_dir: Path,
        dataset: str,
        split: str,
        profile: str,
        inference: InferenceBatch,
        evaluation: EvaluationBatch,
        budget: BudgetContract,
        binding: ProvenanceBinding,
        resolved_config: Mapping[str, Any],
        index_metrics: Mapping[str, Any] | None = None,
        resume: bool = True,
        max_questions: int | None = None,
        question_workers: int = 1,
        question_indices: Sequence[int] | None = None,
    ) -> RunResult:
        validate_protocol_budget(
            self.protocol,
            dataset=dataset,
            profile=profile,
            budget=budget,
        )
        if set(evaluation.by_handle_id) != {
            handle.handle_id for handle in inference.handles
        }:
            raise ExperimentRunError("inference and evaluation handles differ")
        for handle, payload in inference.items():
            if handle.dataset != dataset or payload.dataset != dataset:
                raise ExperimentRunError("run dataset differs from an inference record")
            if handle.split != split:
                raise ExperimentRunError("run split differs from an inference handle")
        if max_questions is not None and max_questions < 0:
            raise ExperimentRunError("max_questions must be non-negative")
        if max_questions is not None and question_indices is not None:
            raise ExperimentRunError(
                "max_questions and question_indices are mutually exclusive"
            )
        normalized_indices = (
            tuple(int(index) for index in question_indices)
            if question_indices is not None
            else None
        )
        if normalized_indices is not None:
            if normalized_indices != tuple(sorted(set(normalized_indices))):
                raise ExperimentRunError(
                    "question_indices must be unique and strictly increasing"
                )
            if any(
                index < 0 or index >= len(inference.handles)
                for index in normalized_indices
            ):
                raise ExperimentRunError("question_indices contains an out-of-range index")
        if question_workers < 1:
            raise ExperimentRunError("question_workers must be positive")
        run_dir = Path(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()) and not resume:
            raise ExperimentRunError("run directory is not empty and resume is disabled")
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "checkpoints" / "records.jsonl"
        if not checkpoint_path.exists():
            atomic_write_jsonl(checkpoint_path, ())
        checkpoints = CheckpointStore(checkpoint_path)
        safe_config = redact(dict(resolved_config))
        if not isinstance(safe_config, dict):
            raise ExperimentRunError("resolved_config must be an object")
        resolved_config_sha256 = hashlib.sha256(canonical_json_bytes(safe_config)).hexdigest()
        if resolved_config_sha256 != binding.resolved_config_sha256:
            raise CheckpointMismatchError("resolved config does not match the provenance binding")
        started_at = datetime.now(timezone.utc).isoformat()
        run_identity = {
            "dataset": dataset,
            "split": split,
            "profile": profile,
            "question_workers": question_workers,
            "binding": binding.sha256(),
        }
        if normalized_indices is not None:
            run_identity["question_indices"] = list(normalized_indices)
        run_id = hashlib.sha256(canonical_json_bytes(run_identity)).hexdigest()[:20]
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            previous_manifest = read_json(manifest_path)
            if not isinstance(previous_manifest, Mapping):
                raise ExperimentRunError("existing run manifest must be an object")
            if previous_manifest.get("binding_sha256") != binding.sha256():
                raise CheckpointMismatchError("run manifest provenance does not match")
            expected_identity = {
                "run_id": run_id,
                "dataset": dataset,
                "split": split,
                "execution_profile": profile,
                "question_workers": question_workers,
            }
            if normalized_indices is not None:
                expected_identity["question_indices"] = list(normalized_indices)
            if any(previous_manifest.get(key) != value for key, value in expected_identity.items()):
                raise CheckpointMismatchError("run manifest identity does not match")
            started_at = str(previous_manifest.get("started_at") or started_at)
        atomic_write_json(run_dir / "resolved_config.json", safe_config)
        llm_calls_path = run_dir / "llm_calls.jsonl"
        if not llm_calls_path.exists():
            atomic_write_jsonl(llm_calls_path, ())

        resumed_count = 0
        all_items = list(inference.items())
        selected_items = all_items
        if normalized_indices is not None:
            selected_items = [all_items[index] for index in normalized_indices]
        if max_questions is not None:
            selected_items = selected_items[:max_questions]
        completed: dict[int, QuestionExecution] = {}
        pending: list[tuple[int, str, InferencePayload, EvaluationAnnotations]] = []
        for index, (handle, payload) in enumerate(selected_items):
            checkpoint_id = f"question:{handle.handle_id}"
            prior = checkpoints.resume(checkpoint_id, binding) if resume else None
            if prior is not None:
                completed[index] = QuestionExecution(
                    prediction=PredictionRecord.model_validate(prior.state["prediction"]),
                    trace=dict(prior.state["trace"]),
                    metrics=QuestionMetrics.model_validate(prior.state["metrics"]),
                    call_records=(),
                )
                resumed_count += 1
                continue
            pending.append(
                (
                    index,
                    handle.handle_id,
                    payload,
                    evaluation.by_handle_id[handle.handle_id],
                )
            )

        def commit(index: int, handle_id: str, execution: QuestionExecution) -> None:
            checkpoint_state = {
                "prediction": execution.prediction.model_dump(mode="json"),
                "trace": execution.trace,
                "metrics": execution.metrics.model_dump(mode="json"),
                "question_workers": question_workers,
            }
            checkpoints.append(
                checkpoint_id=f"question:{handle_id}",
                stage="question_complete",
                binding=binding,
                state=checkpoint_state,
            )
            for call_record in execution.call_records:
                append_jsonl(llm_calls_path, call_record)
            completed[index] = execution

        def execute(
            handle_id: str,
            payload: InferencePayload,
            annotations: EvaluationAnnotations,
        ) -> QuestionExecution:
            question_started = time.perf_counter()
            try:
                return self._process_question(
                    handle_id=handle_id,
                    payload=payload,
                    annotations=annotations,
                    budget=budget,
                )
            except Exception as exc:
                return self._failed_question(
                    handle_id=handle_id,
                    annotations=annotations,
                    budget=budget,
                    error=exc,
                    question_started=question_started,
                )

        if question_workers == 1:
            for index, handle_id, payload, annotations in pending:
                execution = execute(handle_id, payload, annotations)
                commit(index, handle_id, execution)
        elif pending:
            with ThreadPoolExecutor(
                max_workers=min(question_workers, len(pending)),
                thread_name_prefix="eager-question",
            ) as executor:
                futures = {
                    executor.submit(
                        execute,
                        handle_id,
                        payload,
                        annotations,
                    ): (index, handle_id)
                    for index, handle_id, payload, annotations in pending
                }
                for future in as_completed(futures):
                    index, handle_id = futures[future]
                    try:
                        execution = future.result()
                    except Exception as exc:
                        raise ExperimentRunError(
                            f"question worker failed outside the execution boundary: {exc}"
                        ) from exc
                    else:
                        commit(index, handle_id, execution)

        ordered = [completed[index] for index in sorted(completed)]
        predictions = [item.prediction for item in ordered]
        traces = [item.trace for item in ordered]
        question_metrics = [item.metrics for item in ordered]

        atomic_write_jsonl(
            run_dir / "predictions.jsonl",
            (prediction.model_dump(mode="json") for prediction in predictions),
        )
        atomic_write_jsonl(run_dir / "retrieval_traces.jsonl", traces)
        aggregate = _aggregate(question_metrics)
        aggregate["index_build_time_and_size"] = (
            redact(dict(index_metrics)) if index_metrics is not None else None
        )
        atomic_write_json(
            run_dir / "metrics.json",
            {
                "aggregate": aggregate,
                "questions": [row.model_dump(mode="json") for row in question_metrics],
            },
        )
        atomic_write_text(run_dir / "report.md", _report(aggregate))
        expected_question_count = (
            len(selected_items)
            if normalized_indices is not None
            else len(inference.handles)
        )
        complete = len(predictions) == expected_question_count
        finished_at = datetime.now(timezone.utc).isoformat()
        manifest_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "complete" if complete else "partial",
            "started_at": started_at,
            "finished_at": finished_at,
            "dataset": dataset,
            "split": split,
            "execution_profile": profile,
            "question_workers": question_workers,
            "question_indices": (
                list(normalized_indices) if normalized_indices is not None else None
            ),
            "question_count": len(predictions),
            "expected_question_count": expected_question_count,
            "full_dataset_question_count": len(inference.handles),
            "protocol_sha256": protocol_sha256(self.protocol),
            "binding_sha256": binding.sha256(),
            "budget": budget.model_dump(mode="json"),
            "git": dict(self.git_snapshotter(self.repo_root)),
            "environment": dict(self.environment_snapshotter()),
            "index_metrics": redact(dict(index_metrics)) if index_metrics is not None else None,
            "outputs": [
                "resolved_config.json",
                "predictions.jsonl",
                "retrieval_traces.jsonl",
                "llm_calls.jsonl",
                "metrics.json",
                "report.md",
                "checkpoints/records.jsonl",
            ],
        }
        write_manifest(run_dir / "run_manifest.json", manifest_payload)
        inventory_paths = [
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path.name != "output_inventory.json"
        ]
        inventory = output_inventory(run_dir, paths=inventory_paths)
        atomic_write_json(run_dir / "output_inventory.json", inventory)
        return RunResult(
            run_dir=run_dir,
            prediction_count=len(predictions),
            resumed_count=resumed_count,
            status="complete" if complete else "partial",
            metrics=aggregate,
        )


__all__ = [
    "ExperimentRunError",
    "ExperimentRunner",
    "AnswerJudge",
    "AutoPolicyExecutor",
    "FixedPolicyExecutor",
    "QuestionMetrics",
    "RetrievalExecutor",
    "RunResult",
    "validate_protocol_budget",
]
