from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from eager.answering import AnswerGeneration, AnswerGenerationError, AnswerRequest
from eager.config import canonical_json_bytes
from eager.data import EvaluationBatch, InferenceBatch
from eager.experiments import ExperimentRunError, ExperimentRunner, FixedPolicyExecutor
from eager.evaluation import JudgeEvaluation
from eager.indexing import build_tiny_index, load_corpus_snapshot
from eager.pipeline import RetrievalEngine
from eager.protocol import load_protocol
from eager.retrieval import RetrievalKernel, RetrievalKernelConfig, TinySemanticIndex
from eager.runtime.checkpoint import CheckpointMismatchError, ProvenanceBinding
from eager.runtime.io import append_jsonl, read_json, read_jsonl
from eager.runtime.logging import redact
from eager.schemas import (
    BudgetContract,
    EvaluationAnnotations,
    FinalOperation,
    GeneratedSignalStep,
    InferencePayload,
    PredictionRecord,
    QuestionHandle,
)
from eager.signals import (
    ChainGeneration,
    RetrievalQuestionController,
    RetrievalQuestionGeneration,
    StaticIntermediateAnswerer,
    SubquestionChainController,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "indexing" / "corpus.jsonl"
QUESTION = "Where is the company founded by Mira Sol headquartered?"
OUTPUT_FILES = {
    "checkpoints/records.jsonl",
    "llm_calls.jsonl",
    "metrics.json",
    "output_inventory.json",
    "predictions.jsonl",
    "report.md",
    "resolved_config.json",
    "retrieval_traces.jsonl",
    "run_manifest.json",
}


class RecordingAnswerGenerator:
    def __init__(self) -> None:
        self.requests: list[AnswerRequest] = []

    def generate(self, request: AnswerRequest) -> str:
        self.requests.append(request)
        return "Harbor City"


class PartiallyFailingAnswerGenerator:
    def generate_audited(self, request: AnswerRequest) -> AnswerGeneration:
        if request.payload.question.startswith("fail"):
            raise AnswerGenerationError(
                "answer response does not contain a non-empty answer",
                llm_calls=2,
                input_tokens=20,
                output_tokens=4,
                call_records=(
                    {"kind": "answer_generation", "format_attempt": 1},
                    {"kind": "answer_generation", "format_attempt": 2},
                ),
            )
        return AnswerGeneration(answer="Harbor City")


class CountingRetrieval:
    def __init__(self, inner: FixedPolicyExecutor) -> None:
        self.inner = inner
        self.payloads: list[InferencePayload] = []

    def run(self, *, payload: InferencePayload, budget: BudgetContract):
        self.payloads.append(payload)
        return self.inner.run(payload=payload, budget=budget)


class SchedulingProbeRetrieval:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.completion_order: list[str] = []

    def run(self, *, payload: InferencePayload, budget: BudgetContract):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.08 if payload.question.startswith("slow") else 0.01)
            return self.inner.run(payload=payload, budget=budget)
        finally:
            with self.lock:
                self.active -= 1
                self.completion_order.append(payload.question)


def _stack(tmp_path: Path) -> tuple[RetrievalEngine, tuple[str, ...], tuple[str, ...]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS_FIXTURE.read_bytes())
    corpus = load_corpus_snapshot(corpus_path)
    index_dir = tmp_path / "index"
    build_tiny_index(corpus, index_dir, dimensions=32)
    semantic = TinySemanticIndex.load(index_dir, corpus=corpus)
    engine = RetrievalEngine(
        RetrievalKernel(
            corpus=corpus,
            semantic_index=semantic,
            config=RetrievalKernelConfig(
                candidate_limit=50,
                paragraph_limit=5,
                rerank_limit=50,
                evidence_limit=5,
            ),
        )
    )
    sentence_ids = tuple(item.sentence_id for item in corpus.evidence_units)
    paragraph_ids = tuple(dict.fromkeys(item.paragraph_id for item in corpus.evidence_units))
    return engine, sentence_ids, paragraph_ids


def _batches(
    sentence_ids: tuple[str, ...],
    paragraph_ids: tuple[str, ...],
    *,
    sentence_gold_for_second: bool = True,
) -> tuple[InferenceBatch, EvaluationBatch]:
    handles = tuple(
        QuestionHandle.create(dataset="2wiki", split="fixture", raw_dataset_id=f"q{index}")
        for index in (1, 2)
    )
    inference = InferenceBatch(
        handles=handles,
        payloads=tuple(
            InferencePayload(dataset="2wiki", question=QUESTION) for _handle in handles
        ),
    )
    annotations = {
        handle.handle_id: EvaluationAnnotations(
            handle_id=handle.handle_id,
            answer="Harbor City",
            supporting_sentence_ids=(
                sentence_ids if index == 0 or sentence_gold_for_second else ()
            ),
            supporting_paragraph_ids=paragraph_ids,
        )
        for index, handle in enumerate(handles)
    }
    return inference, EvaluationBatch(by_handle_id=annotations)


def _budget(**overrides: int) -> BudgetContract:
    values = {
        "max_iterations": 2,
        "max_candidates_per_step": 50,
        "max_candidates_per_question": 100,
        "max_parent_units_per_step": 5,
        "max_parent_units_per_question": 10,
        "max_rerank_pairs_per_step": 50,
        "max_rerank_pairs_per_question": 100,
        "max_retained_evidence": 5,
        "max_llm_calls_per_step": 0,
        "max_llm_calls_per_question": 0,
        "max_input_tokens_per_question": 0,
        "max_output_tokens_per_question": 0,
    }
    return BudgetContract(**{**values, **overrides})


def _binding(config: dict[str, Any], *, data: str = "data") -> ProvenanceBinding:
    public = redact(config)
    assert isinstance(public, dict)
    return ProvenanceBinding(
        resolved_config_sha256=hashlib.sha256(canonical_json_bytes(public)).hexdigest(),
        data_sha256=data,
        index_sha256="index",
        model_revision="tiny-v1",
        prompt_sha256="prompt",
    )


def _runner(
    tmp_path: Path,
    *,
    answer_judge=None,
    controller_factory=None,
    final_answer_scope: str = "global",
):
    engine, sentence_ids, paragraph_ids = _stack(tmp_path / "stack")
    factory = controller_factory or (lambda: RetrievalQuestionController(max_steps=2))
    retrieval = CountingRetrieval(
        FixedPolicyExecutor(
            engine,
            factory,
        )
    )
    answer = RecordingAnswerGenerator()
    runner = ExperimentRunner(
        retrieval=retrieval,
        answer_generator=answer,
        protocol=load_protocol(REPO_ROOT / "wsdm" / "config" / "experiment_protocol.json"),
        repo_root=REPO_ROOT,
        answer_judge=answer_judge,
        final_answer_scope=final_answer_scope,
        git_snapshotter=lambda _root: {"commit": "a" * 40, "branch": "test", "dirty": False},
        environment_snapshotter=lambda: {"python_version": "test"},
    )
    return runner, retrieval, answer, sentence_ids, paragraph_ids


class AuditedCorrectJudge:
    def judge_audited(self, *, prediction: str, annotations: EvaluationAnnotations):
        assert prediction == annotations.answer
        return JudgeEvaluation(
            correct=True,
            llm_calls=1,
            input_tokens=8,
            output_tokens=1,
            call_record={
                "kind": "gpt_accuracy_judge",
                "response_sha256": "c" * 64,
                "input_tokens": 8,
                "output_tokens": 1,
            },
        )


class AuditedSignalGenerator:
    def generate(self, payload, retrieval_state, iteration):
        return RetrievalQuestionGeneration(
            text=payload.question,
            llm_calls=1,
            input_tokens=7,
            output_tokens=2,
            call_records=(
                {
                    "kind": "retrieval_question_generation",
                    "response_sha256": "d" * 64,
                    "input_tokens": 7,
                    "output_tokens": 2,
                    "iteration": iteration,
                },
            ),
        )


def test_complete_run_writes_exact_schema_bound_artifacts(tmp_path: Path) -> None:
    runner, retrieval, answer, sentence_ids, paragraph_ids = _runner(tmp_path)
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    config = {"method": "fixture", "nested": {"api_key": "private-value"}}
    run_dir = tmp_path / "run"

    result = runner.run(
        run_dir=run_dir,
        dataset="2wiki",
        split="fixture",
        profile="wsdm_strict",
        inference=inference,
        evaluation=evaluation,
        budget=_budget(),
        binding=_binding(config),
        resolved_config=config,
        index_metrics={"build_seconds": 1.25, "bytes": 1234},
    )

    assert result.status == "complete"
    assert result.prediction_count == 2
    assert result.resumed_count == 0
    assert len(retrieval.payloads) == len(answer.requests) == 2
    assert {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()} == OUTPUT_FILES
    assert "private-value" not in (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    assert read_json(run_dir / "resolved_config.json")["nested"]["api_key"] == "<redacted>"

    predictions = read_jsonl(run_dir / "predictions.jsonl")
    assert len(predictions) == 2
    for row in predictions:
        PredictionRecord.model_validate(row)
        assert row["answer"] == "Harbor City"
        assert set(row["budget"]) == set(_budget().model_dump(mode="json"))

    metrics = read_json(run_dir / "metrics.json")
    aggregate = metrics["aggregate"]
    assert aggregate["question_count"] == 2
    assert aggregate["normalized_em"] == 1.0
    assert aggregate["candidate_count_mean_p50_p95"]["p95"] >= 0
    assert aggregate["full_support_at_5_10_20"]["sentence"]["5"]["applicable_question_count"] == 2
    assert aggregate["index_build_time_and_size"] == {"build_seconds": 1.25, "bytes": 1234}
    assert aggregate["gpt_accuracy_status"] == "judge_not_configured"
    assert aggregate["controller_chain"] == {
        "planner_call_count": 0,
        "planner_retry_count": 0,
        "planner_rejected_count": 0,
        "intermediate_answer_retry_count": 0,
        "failed_step_count": 0,
        "blocked_step_count": 0,
    }

    inventory = read_json(run_dir / "output_inventory.json")
    inventory_paths = {row["path"] for row in inventory}
    assert inventory_paths == OUTPUT_FILES - {"output_inventory.json"}
    for row in inventory:
        path = run_dir / row["path"]
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_runner_forwards_final_operation_to_existing_answer_generator(
    tmp_path: Path,
) -> None:
    operation = FinalOperation(
        type="return",
        inputs=("step_1",),
        instruction="Return the location supported by step 1.",
    )

    class OperationGenerator:
        def generate(self, payload, _max_steps):
            return ChainGeneration(
                steps=(
                    GeneratedSignalStep(
                        step_id="step_1",
                        text=payload.question,
                    ),
                ),
                final_operation=operation,
            )

    runner, _retrieval, answer, sentence_ids, paragraph_ids = _runner(
        tmp_path,
        controller_factory=lambda: SubquestionChainController(
            generator=OperationGenerator(),
        ),
    )
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    config = {"method": "dag-operation"}

    runner.run(
        run_dir=tmp_path / "run",
        dataset="2wiki",
        split="fixture",
        profile="wsdm_strict",
        inference=inference,
        evaluation=evaluation,
        budget=_budget(),
        binding=_binding(config),
        resolved_config=config,
    )

    assert len(answer.requests) == 2
    assert all(request.final_operation == operation for request in answer.requests)


def test_last_subquestion_scope_uses_terminal_question_and_local_evidence(
    tmp_path: Path,
) -> None:
    class LinearGenerator:
        def generate(self, payload, _max_steps):
            return (
                GeneratedSignalStep(step_id="step_1", text="Find the company."),
                GeneratedSignalStep(
                    step_id="step_2",
                    text="Where is #1 headquartered?",
                    depends_on=("step_1",),
                ),
            )

    runner, _retrieval, answer, sentence_ids, paragraph_ids = _runner(
        tmp_path,
        final_answer_scope="last_subquestion",
        controller_factory=lambda: SubquestionChainController(
            generator=LinearGenerator(),
            answerer=StaticIntermediateAnswerer({"step_1": "Harbor City"}),
            final_answerer=StaticIntermediateAnswerer(
                {"step_1": "Harbor City", "step_2": "Harbor City"}
            ),
        ),
    )
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    config = {"method": "last-subquestion-answer"}

    runner.run(
        run_dir=tmp_path / "run",
        dataset="2wiki",
        split="fixture",
        profile="wsdm_strict",
        inference=inference,
        evaluation=evaluation,
        budget=_budget(),
        binding=_binding(config),
        resolved_config=config,
    )

    assert answer.requests == []


def test_partial_resume_is_idempotent_and_preserves_run_state(tmp_path: Path) -> None:
    runner, retrieval, _answer, sentence_ids, paragraph_ids = _runner(tmp_path)
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    config = {"method": "resume"}
    run_dir = tmp_path / "run"
    arguments = {
        "run_dir": run_dir,
        "dataset": "2wiki",
        "split": "fixture",
        "profile": "wsdm_strict",
        "inference": inference,
        "evaluation": evaluation,
        "budget": _budget(),
        "binding": _binding(config),
        "resolved_config": config,
    }

    first = runner.run(**arguments, max_questions=1)
    started_at = read_json(run_dir / "run_manifest.json")["started_at"]
    append_jsonl(run_dir / "llm_calls.jsonl", {"call_id": "preserved"})
    second = runner.run(**arguments)

    assert first.status == "partial"
    assert second.status == "complete"
    assert second.resumed_count == 1
    assert len(retrieval.payloads) == 2
    assert read_json(run_dir / "run_manifest.json")["started_at"] == started_at
    assert read_jsonl(run_dir / "llm_calls.jsonl") == [{"call_id": "preserved"}]
    assert len(read_jsonl(run_dir / "checkpoints" / "records.jsonl")) == 2


def test_runner_rejects_provenance_and_protocol_budget_mismatches(tmp_path: Path) -> None:
    runner, _retrieval, _answer, sentence_ids, paragraph_ids = _runner(tmp_path)
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    config = {"method": "bound"}
    run_dir = tmp_path / "run"
    common = {
        "run_dir": run_dir,
        "dataset": "2wiki",
        "split": "fixture",
        "profile": "wsdm_strict",
        "inference": inference,
        "evaluation": evaluation,
        "budget": _budget(),
        "binding": _binding(config),
        "resolved_config": config,
    }
    runner.run(**common, max_questions=1)

    changed = {**common, "binding": _binding({"method": "changed"}, data="other"), "resolved_config": {"method": "changed"}}
    with pytest.raises(CheckpointMismatchError, match="provenance"):
        runner.run(**changed)
    with pytest.raises(ExperimentRunError, match="candidate step budget"):
        runner.run(**{**common, "run_dir": tmp_path / "bad-candidates", "budget": _budget(max_candidates_per_step=51)})
    with pytest.raises(ExperimentRunError, match="retained evidence budget"):
        runner.run(**{**common, "run_dir": tmp_path / "bad-evidence", "budget": _budget(max_retained_evidence=6)})


def test_missing_support_granularity_is_excluded_and_annotations_are_isolated(
    tmp_path: Path,
) -> None:
    runner, retrieval, answer, sentence_ids, paragraph_ids = _runner(tmp_path)
    inference, evaluation = _batches(
        sentence_ids,
        paragraph_ids,
        sentence_gold_for_second=False,
    )
    config = {"method": "isolation"}
    result = runner.run(
        run_dir=tmp_path / "run",
        dataset="2wiki",
        split="fixture",
        profile="wsdm_strict",
        inference=inference,
        evaluation=evaluation,
        budget=_budget(),
        binding=_binding(config),
        resolved_config=config,
    )

    sentence_rate = result.metrics["sentence_full_support_recall"]
    assert sentence_rate["applicable_question_count"] == 1
    assert result.metrics["paragraph_full_support_recall"]["applicable_question_count"] == 2
    assert [payload.model_dump(mode="json") for payload in retrieval.payloads] == [
        {"dataset": "2wiki", "question": QUESTION},
        {"dataset": "2wiki", "question": QUESTION},
    ]
    assert all(
        request.payload.model_dump(mode="json") == {"dataset": "2wiki", "question": QUESTION}
        for request in answer.requests
    )


def test_audited_judge_contributes_gpt_accuracy_usage_and_call_log(tmp_path: Path) -> None:
    runner, _retrieval, _answer, sentence_ids, paragraph_ids = _runner(
        tmp_path,
        answer_judge=AuditedCorrectJudge(),
    )
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    config = {"method": "judged"}
    result = runner.run(
        run_dir=tmp_path / "run",
        dataset="2wiki",
        split="fixture",
        profile="wsdm_strict",
        inference=inference,
        evaluation=evaluation,
        budget=_budget(
            max_llm_calls_per_question=1,
            max_input_tokens_per_question=16,
            max_output_tokens_per_question=4,
        ),
        binding=_binding(config),
        resolved_config=config,
    )

    assert result.metrics["gpt_accuracy"] == 1.0
    assert result.metrics["gpt_accuracy_status"] == "complete"
    assert result.metrics["llm_calls_and_tokens"] == {
        "calls": 2,
        "input_tokens": 16,
        "output_tokens": 2,
    }
    calls = read_jsonl(tmp_path / "run" / "llm_calls.jsonl")
    assert len(calls) == 2
    assert all(call["kind"] == "gpt_accuracy_judge" for call in calls)


def test_signal_usage_and_call_records_are_included_in_run_outputs(tmp_path: Path) -> None:
    runner, _retrieval, _answer, sentence_ids, paragraph_ids = _runner(
        tmp_path,
        controller_factory=lambda: RetrievalQuestionController(
            generator=AuditedSignalGenerator(),
            max_steps=2,
        ),
    )
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    config = {"method": "audited-signal"}
    result = runner.run(
        run_dir=tmp_path / "run",
        dataset="2wiki",
        split="fixture",
        profile="wsdm_strict",
        inference=inference,
        evaluation=evaluation,
        budget=_budget(
            max_llm_calls_per_step=1,
            max_llm_calls_per_question=1,
            max_input_tokens_per_question=10,
            max_output_tokens_per_question=5,
        ),
        binding=_binding(config),
        resolved_config=config,
    )

    assert result.metrics["llm_calls_and_tokens"] == {
        "calls": 2,
        "input_tokens": 14,
        "output_tokens": 4,
    }
    calls = read_jsonl(tmp_path / "run" / "llm_calls.jsonl")
    assert len(calls) == 2
    assert all(call["kind"] == "retrieval_question_generation" for call in calls)
    assert all(call["retrieval_stage"] == "signal" for call in calls)
    assert all(call["iteration"] == 2 for call in calls)


def test_question_workers_schedule_dynamically_and_preserve_output_order(tmp_path: Path) -> None:
    runner, retrieval, _answer, sentence_ids, paragraph_ids = _runner(tmp_path)
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    inference = InferenceBatch(
        handles=inference.handles,
        payloads=(
            InferencePayload(dataset="2wiki", question=f"slow {QUESTION}"),
            InferencePayload(dataset="2wiki", question=f"fast {QUESTION}"),
        ),
    )
    probe = SchedulingProbeRetrieval(retrieval)
    runner.retrieval = probe
    config = {"method": "concurrent"}
    arguments = {
        "run_dir": tmp_path / "run",
        "dataset": "2wiki",
        "split": "fixture",
        "profile": "wsdm_strict",
        "inference": inference,
        "evaluation": evaluation,
        "budget": _budget(),
        "binding": _binding(config),
        "resolved_config": config,
        "question_workers": 2,
    }

    result = runner.run(**arguments)

    assert result.status == "complete"
    assert probe.max_active == 2
    assert probe.completion_order[0].startswith("fast")
    assert [row["handle_id"] for row in read_jsonl(tmp_path / "run" / "predictions.jsonl")] == [
        handle.handle_id for handle in inference.handles
    ]
    assert read_json(tmp_path / "run" / "run_manifest.json")["question_workers"] == 2
    assert runner.run(**arguments).resumed_count == 2
    with pytest.raises(CheckpointMismatchError, match="identity"):
        runner.run(**{**arguments, "question_workers": 1})


def test_question_error_is_checkpointed_counted_as_wrong_and_does_not_stop_peers(
    tmp_path: Path,
) -> None:
    runner, _retrieval, _answer, sentence_ids, paragraph_ids = _runner(tmp_path)
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    inference = InferenceBatch(
        handles=inference.handles,
        payloads=(
            InferencePayload(dataset="2wiki", question=f"ok {QUESTION}"),
            InferencePayload(dataset="2wiki", question=f"fail {QUESTION}"),
        ),
    )
    runner.answer_generator = PartiallyFailingAnswerGenerator()
    config = {"method": "recover-question-error"}
    run_dir = tmp_path / "run"

    result = runner.run(
        run_dir=run_dir,
        dataset="2wiki",
        split="fixture",
        profile="wsdm_strict",
        inference=inference,
        evaluation=evaluation,
        budget=_budget(
            max_llm_calls_per_question=2,
            max_input_tokens_per_question=20,
            max_output_tokens_per_question=4,
        ),
        binding=_binding(config),
        resolved_config=config,
        question_workers=2,
    )

    assert result.status == "complete"
    assert result.prediction_count == 2
    assert result.metrics["successful_question_count"] == 1
    assert result.metrics["error_question_count"] == 1
    assert result.metrics["error_rate"] == 0.5
    assert result.metrics["normalized_em"] == 0.5
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    assert [row["status"] for row in predictions] == ["ok", "error"]
    assert predictions[1]["answer"] is None
    assert predictions[1]["error"]["code"] == "answer_format_error"
    assert len(read_jsonl(run_dir / "llm_calls.jsonl")) == 2
    assert len(read_jsonl(run_dir / "checkpoints" / "records.jsonl")) == 2


def test_sparse_question_indices_are_bound_and_complete(tmp_path: Path) -> None:
    runner, retrieval, _answer, sentence_ids, paragraph_ids = _runner(tmp_path)
    inference, evaluation = _batches(sentence_ids, paragraph_ids)
    config = {"method": "sparse-selection", "question_indices": [1]}
    run_dir = tmp_path / "run"
    common = {
        "run_dir": run_dir,
        "dataset": "2wiki",
        "split": "fixture",
        "profile": "wsdm_strict",
        "inference": inference,
        "evaluation": evaluation,
        "budget": _budget(),
        "binding": _binding(config),
        "resolved_config": config,
        "question_indices": (1,),
    }

    result = runner.run(**common)

    assert result.status == "complete"
    assert result.prediction_count == 1
    assert len(retrieval.payloads) == 1
    assert read_jsonl(run_dir / "predictions.jsonl")[0]["handle_id"] == (
        inference.handles[1].handle_id
    )
    manifest = read_json(run_dir / "run_manifest.json")
    assert manifest["question_indices"] == [1]
    assert manifest["expected_question_count"] == 1
    assert manifest["full_dataset_question_count"] == 2
    assert runner.run(**common).resumed_count == 1

    with pytest.raises(ExperimentRunError, match="mutually exclusive"):
        runner.run(**{**common, "max_questions": 1})
    with pytest.raises(ExperimentRunError, match="strictly increasing"):
        runner.run(**{**common, "question_indices": (1, 1)})
    with pytest.raises(ExperimentRunError, match="out-of-range"):
        runner.run(**{**common, "question_indices": (2,)})
