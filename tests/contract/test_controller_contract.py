from __future__ import annotations

import pytest

from eager.schemas import (
    BudgetContract,
    BudgetExceededError,
    ControllerState,
    EvidenceCandidate,
    EvidenceUnit,
    InferencePayload,
    QuestionHandle,
    RetrievalState,
    SignalDecision,
    StepBudgetUsage,
    StepOutcome,
)
from eager.signals import ControllerContractError, SignalController


def budget() -> BudgetContract:
    return BudgetContract(
        max_iterations=2,
        max_candidates_per_step=2,
        max_candidates_per_question=4,
        max_parent_units_per_step=2,
        max_parent_units_per_question=4,
        max_rerank_pairs_per_step=2,
        max_rerank_pairs_per_question=4,
        max_retained_evidence=2,
        max_llm_calls_per_step=1,
        max_llm_calls_per_question=2,
        max_input_tokens_per_question=100,
        max_output_tokens_per_question=100,
    )


def unit(iteration: int) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=f"e{iteration}",
        paragraph_id=f"p{iteration}",
        title="Title",
        sentence_id=f"s{iteration}",
        text=f"Evidence {iteration}",
    )


class ScriptedController(SignalController):
    @property
    def controller_type(self) -> str:
        return "scripted"

    def _start(self, payload: InferencePayload, budget: BudgetContract) -> ControllerState:
        return ControllerState.started(controller_type=self.controller_type, budget=budget)

    def _produce_signal(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
    ) -> SignalDecision:
        iteration = state.completed_steps + 1
        return SignalDecision(
            decision_id=f"signal-{iteration}",
            controller_type=self.controller_type,
            iteration=iteration,
            action="retrieve",
            reason="scripted step",
            signal_text=f"retrieve step {iteration}",
        )

    def _consume_evidence(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
        decision: SignalDecision,
    ) -> StepOutcome:
        is_final = decision.iteration == state.budget.max_iterations
        return StepOutcome.create(
            state=state,
            decision=decision,
            consumed_evidence_ids=[retrieval_state.retained_evidence[-1].evidence_id],
            next_action="stop" if is_final else "continue",
            reason="budget reached" if is_final else "more evidence needed",
        )


def advance(state: RetrievalState, decision: SignalDecision) -> RetrievalState:
    evidence = unit(decision.iteration)
    return state.advance(
        signal=decision,
        anchors=[],
        candidates=[EvidenceCandidate(evidence=evidence, sources=["entity"], coarse_score=1.0)],
        new_evidence=[evidence],
        step_usage=StepBudgetUsage(candidates_considered=1, parent_units_selected=1, rerank_pairs=1),
    )


def test_controller_enforces_full_lifecycle_and_iteration_sync() -> None:
    controller = ScriptedController()
    payload = InferencePayload(dataset="2wiki", question="Who wrote it?")
    retrieval = RetrievalState.initial(budget())
    state = controller.start(payload, retrieval.budget)

    first = controller.produce_signal(payload, retrieval, state)
    assert first.state.phase == "awaiting_evidence"
    retrieval = advance(retrieval, first.decision)
    first_outcome = controller.consume_evidence(payload, retrieval, first.state, first.decision)
    assert first_outcome.state.phase == "ready"

    second = controller.produce_signal(payload, retrieval, first_outcome.state)
    retrieval = advance(retrieval, second.decision)
    second_outcome = controller.consume_evidence(payload, retrieval, second.state, second.decision)
    assert second_outcome.state.phase == "stopped"
    assert second_outcome.state.completed_steps == 2
    assert second_outcome.state.consumed_signal_ids == ("signal-1", "signal-2")


def test_controller_rejects_orchestration_handle_instead_of_inference_payload() -> None:
    controller = ScriptedController()
    handle = QuestionHandle.create(dataset="musique", split="dev", raw_dataset_id="2hop__1")

    with pytest.raises(TypeError, match="only InferencePayload"):
        controller.start(handle, budget())  # type: ignore[arg-type]


def test_controller_rejects_out_of_sync_retrieval_state() -> None:
    controller = ScriptedController()
    payload = InferencePayload(dataset="2wiki", question="Question")
    retrieval = RetrievalState.initial(budget())
    state = controller.start(payload, retrieval.budget)
    emission = controller.produce_signal(payload, retrieval, state)

    with pytest.raises(ControllerContractError, match="requires ready state"):
        controller.produce_signal(payload, retrieval, emission.state)


def test_controller_cannot_produce_after_iteration_budget() -> None:
    controller = ScriptedController()
    payload = InferencePayload(dataset="2wiki", question="Question")
    contract = budget()
    exhausted = ControllerState(
        controller_type="scripted",
        budget=contract,
        completed_steps=2,
        consumed_signal_ids=("signal-1", "signal-2"),
    )
    retrieval = RetrievalState(
        iteration=2,
        budget=contract,
        total_usage={"iterations": 2},
    )

    with pytest.raises(BudgetExceededError, match="exhausted"):
        controller.produce_signal(payload, retrieval, exhausted)
