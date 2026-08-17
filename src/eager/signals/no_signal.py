from __future__ import annotations

from eager.schemas import (
    BudgetContract,
    ControllerState,
    InferencePayload,
    RetrievalState,
    SignalDecision,
    StepOutcome,
)
from eager.signals._identity import decision_id
from eager.signals.base import SignalController


class NoSignalController(SignalController):
    def __init__(self, *, max_steps: int = 1) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps

    @property
    def controller_type(self) -> str:
        return "no_signal"

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
            decision_id=decision_id(self.controller_type, iteration, payload.question),
            controller_type=self.controller_type,
            iteration=iteration,
            action="retrieve",
            reason="use the original question without a generated step signal",
            signal_text=payload.question,
            selected_by="fixed_no_signal_policy",
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
            else "fixed no-signal step limit reached"
            if exhausted
            else "continue fixed no-signal retrieval"
        )
        return StepOutcome.create(
            state=state,
            decision=decision,
            consumed_evidence_ids=consumed,
            next_action=next_action,
            reason=reason,
        )


__all__ = ["NoSignalController"]
