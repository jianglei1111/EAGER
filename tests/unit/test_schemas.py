from __future__ import annotations

import pytest
from pydantic import ValidationError

from eager.schemas import (
    BudgetContract,
    BudgetExceededError,
    EvaluationAnnotations,
    EvidenceCandidate,
    EvidenceUnit,
    InferencePayload,
    QueryAnchor,
    QuestionHandle,
    RetrievalState,
    SignalDecision,
    StepBudgetUsage,
)


def budget(**overrides: int) -> BudgetContract:
    values = {
        "max_iterations": 2,
        "max_candidates_per_step": 3,
        "max_candidates_per_question": 6,
        "max_parent_units_per_step": 2,
        "max_parent_units_per_question": 4,
        "max_rerank_pairs_per_step": 3,
        "max_rerank_pairs_per_question": 6,
        "max_retained_evidence": 2,
        "max_llm_calls_per_step": 2,
        "max_llm_calls_per_question": 4,
        "max_input_tokens_per_question": 1000,
        "max_output_tokens_per_question": 500,
    }
    values.update(overrides)
    return BudgetContract(**values)


def evidence(index: int, *, text: str | None = None) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=f"e{index}",
        paragraph_id=f"p{index}",
        title=f"Title {index}",
        sentence_id=f"s{index}",
        text=text or f"Evidence {index}",
    )


def candidate(unit: EvidenceUnit) -> EvidenceCandidate:
    return EvidenceCandidate(evidence=unit, sources=["entity"], coarse_score=0.5)


def decision(iteration: int) -> SignalDecision:
    return SignalDecision(
        decision_id=f"d{iteration}",
        controller_type="test",
        iteration=iteration,
        action="retrieve",
        reason="continue retrieval",
        signal_text=f"signal {iteration}",
    )


def test_question_handle_is_deterministic_and_does_not_expose_raw_id() -> None:
    first = QuestionHandle.create(dataset="MuSiQue", split="Dev", raw_dataset_id="2hop__123_456")
    second = QuestionHandle.create(dataset="musique", split="dev", raw_dataset_id="2hop__123_456")

    assert first == second
    assert first.handle_id.startswith("qh_")
    assert first.raw_dataset_id not in first.handle_id


@pytest.mark.parametrize(
    "forbidden",
    [
        "question_id",
        "raw_question_id",
        "answer",
        "supporting_facts",
        "gold_type",
        "hop_count",
        "question_decomposition",
        "relation",
    ],
)
def test_inference_payload_rejects_identifiers_and_annotations(forbidden: str) -> None:
    with pytest.raises(ValidationError, match="forbidden"):
        InferencePayload.model_validate(
            {"dataset": "2wiki", "question": "Who wrote it?", forbidden: "oracle"}
        )


def test_inference_payload_is_frozen_and_serializes_only_allowlisted_fields() -> None:
    payload = InferencePayload(dataset="HotpotQA", question="Who wrote the novel?")

    assert payload.model_dump(mode="json") == {
        "dataset": "hotpotqa",
        "question": "Who wrote the novel?",
    }
    with pytest.raises(ValidationError):
        payload.question = "changed"  # type: ignore[misc]


def test_annotations_stably_deduplicate_evaluator_only_values() -> None:
    handle = QuestionHandle.create(dataset="musique", split="dev", raw_dataset_id="raw-1")
    annotations = EvaluationAnnotations(
        handle_id=handle.handle_id,
        answer="Paris",
        answer_aliases=["Paris", "PARIS", "City of Paris"],
        supporting_sentence_ids=["s2", "s1", "s2"],
        supporting_paragraph_ids=["p2", "p1", "p2"],
        gold_relations=["located in", "located in"],
        hop_count=2,
    )

    assert annotations.answer_aliases == ("Paris", "City of Paris")
    assert annotations.supporting_sentence_ids == ("s2", "s1")
    assert annotations.supporting_paragraph_ids == ("p2", "p1")


def test_query_anchor_factory_has_stable_normalized_identity() -> None:
    first = QueryAnchor.from_surface("  Caf\u00e9  de Flore ", sources=["question"], priority=0)
    second = QueryAnchor.from_surface("CAF\u00c9 DE FLORE", sources=["question"], priority=0)

    assert first.anchor_id == second.anchor_id
    assert first.normalized_surface == "caf\u00e9 de flore"


def test_retrieval_state_uses_old_before_new_stable_dedup_and_cap() -> None:
    state = RetrievalState.initial(budget())
    first = evidence(1)
    second = evidence(2)
    state = state.advance(
        signal=decision(1),
        anchors=[],
        candidates=[candidate(first), candidate(second)],
        new_evidence=[first, second, first],
        step_usage=StepBudgetUsage(candidates_considered=2, parent_units_selected=2, rerank_pairs=2),
    )
    state = state.advance(
        signal=decision(2),
        anchors=[],
        candidates=[candidate(evidence(3))],
        new_evidence=[evidence(3), first],
        step_usage=StepBudgetUsage(candidates_considered=1, parent_units_selected=1, rerank_pairs=1),
    )

    assert [item.evidence_id for item in state.retained_evidence] == ["e1", "e2"]
    assert [item.decision_id for item in state.signal_history] == ["d1", "d2"]


def test_conflicting_stable_evidence_id_is_rejected() -> None:
    state = RetrievalState.initial(budget())
    with pytest.raises(ValueError, match="conflicting records"):
        state.advance(
            signal=decision(1),
            anchors=[],
            candidates=[],
            new_evidence=[evidence(1), evidence(1, text="different")],
            step_usage=StepBudgetUsage(),
        )


def test_budget_is_checked_per_step_and_per_question() -> None:
    state = RetrievalState.initial(budget(max_candidates_per_step=1))
    with pytest.raises(BudgetExceededError, match="step candidates"):
        state.advance(
            signal=decision(1),
            anchors=[],
            candidates=[candidate(evidence(1)), candidate(evidence(2))],
            new_evidence=[],
            step_usage=StepBudgetUsage(candidates_considered=2),
        )


def test_unordered_collections_are_rejected() -> None:
    with pytest.raises(ValidationError, match="ordered sequence"):
        EvaluationAnnotations(
            handle_id=QuestionHandle.create(
                dataset="2wiki", split="dev", raw_dataset_id="raw-2"
            ).handle_id,
            answer="answer",
            supporting_sentence_ids={"s1", "s2"},
        )
