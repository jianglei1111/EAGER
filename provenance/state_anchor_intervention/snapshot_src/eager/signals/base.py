from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, model_validator

from eager.schemas import (
    BudgetContract,
    BudgetExceededError,
    ControllerState,
    InferencePayload,
    RetrievalState,
    SchemaInvariantError,
    SignalDecision,
    StepOutcome,
)


class ControllerContractError(SchemaInvariantError):
    """Raised when a signal-controller implementation violates its contract."""


class SignalEmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: SignalDecision
    state: ControllerState

    @model_validator(mode="after")
    def state_matches_decision(self) -> "SignalEmission":
        if self.state.controller_type != self.decision.controller_type:
            raise ValueError("emission state and decision use different controller types")
        if self.decision.action == "retrieve":
            if self.state.phase != "awaiting_evidence":
                raise ValueError("retrieve emissions must await evidence")
            if self.state.pending_signal_id != self.decision.decision_id:
                raise ValueError("retrieve emissions must retain the pending decision id")
        elif self.state.phase != "stopped":
            raise ValueError("stop emissions must stop the controller")
        return self


class SignalController(ABC):
    """Contract-checked, immutable-state interface for all signal policies.

    Subclasses implement only policy decisions. The public template methods
    enforce the lifecycle ``ready -> awaiting_evidence -> ready|stopped`` and
    keep the controller and retrieval iteration budgets synchronized.
    """

    @property
    @abstractmethod
    def controller_type(self) -> str:
        """Return the stable policy identifier written to traces."""

    def start(self, payload: InferencePayload, budget: BudgetContract) -> ControllerState:
        if not isinstance(payload, InferencePayload):
            raise TypeError("controllers accept only InferencePayload")
        state = self._start(payload, budget)
        if state.controller_type != self.controller_type:
            raise ControllerContractError("start state uses the wrong controller_type")
        if state.budget != budget:
            raise ControllerContractError("start state changed the supplied budget")
        if state.phase != "ready" or state.completed_steps != 0:
            raise ControllerContractError("start must return a fresh ready state")
        return state

    def produce_signal(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
    ) -> SignalEmission:
        if not isinstance(payload, InferencePayload):
            raise TypeError("controllers accept only InferencePayload")
        self._validate_common(retrieval_state, state)
        if state.phase != "ready":
            raise ControllerContractError("produce_signal requires ready state")
        if state.completed_steps >= state.budget.max_iterations:
            raise BudgetExceededError("controller iteration budget is exhausted")
        if retrieval_state.iteration != state.completed_steps:
            raise ControllerContractError("controller and retrieval iterations are out of sync")
        decision = self._produce_signal(payload, retrieval_state, state)
        if decision.controller_type != self.controller_type:
            raise ControllerContractError("decision uses the wrong controller_type")
        next_state = state.begin_signal(decision)
        return SignalEmission(decision=decision, state=next_state)

    def consume_evidence(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
        decision: SignalDecision,
    ) -> StepOutcome:
        if not isinstance(payload, InferencePayload):
            raise TypeError("controllers accept only InferencePayload")
        self._validate_common(retrieval_state, state)
        if state.phase != "awaiting_evidence":
            raise ControllerContractError("consume_evidence requires awaiting_evidence state")
        if state.pending_signal_id != decision.decision_id:
            raise ControllerContractError("consume_evidence received a non-pending decision")
        if decision.action != "retrieve":
            raise ControllerContractError("a stop decision cannot consume evidence")
        if retrieval_state.iteration != state.completed_steps + 1:
            raise ControllerContractError("retrieval must advance exactly once before evidence is consumed")

        outcome = self._consume_evidence(payload, retrieval_state, state, decision)
        if outcome.decision_id != decision.decision_id:
            raise ControllerContractError("outcome consumed the wrong decision")
        if outcome.state.controller_type != self.controller_type:
            raise ControllerContractError("outcome state uses the wrong controller_type")
        if outcome.state.budget != state.budget:
            raise ControllerContractError("outcome state changed the budget")
        if outcome.state.generated_chain != state.generated_chain:
            raise ControllerContractError("generated chains must be frozen after start")
        if outcome.state.final_operation != state.final_operation:
            raise ControllerContractError("final operation must be frozen after start")
        if outcome.state.completed_steps != state.completed_steps + 1:
            raise ControllerContractError("consume_evidence must complete exactly one step")
        if outcome.used_fallback != state.pending_fallback:
            raise ControllerContractError("outcome fallback flag does not match the pending signal")
        retained_ids = {item.evidence_id for item in retrieval_state.retained_evidence}
        unknown_ids = set(outcome.consumed_evidence_ids) - retained_ids
        if unknown_ids:
            raise ControllerContractError(
                f"controller consumed evidence outside the retained state: {sorted(unknown_ids)}"
            )
        if outcome.next_action == "continue" and outcome.state.completed_steps >= state.budget.max_iterations:
            raise BudgetExceededError("controller must stop at the iteration budget")
        return outcome

    def _validate_common(self, retrieval_state: RetrievalState, state: ControllerState) -> None:
        if state.controller_type != self.controller_type:
            raise ControllerContractError("state belongs to a different controller")
        if retrieval_state.budget != state.budget:
            raise ControllerContractError("controller and retrieval states use different budgets")

    @abstractmethod
    def _start(self, payload: InferencePayload, budget: BudgetContract) -> ControllerState:
        """Create controller-private state without reading identifiers or annotations."""

    @abstractmethod
    def _produce_signal(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
    ) -> SignalDecision:
        """Select the next retrieval or stop action from inference-visible state."""

    @abstractmethod
    def _consume_evidence(
        self,
        payload: InferencePayload,
        retrieval_state: RetrievalState,
        state: ControllerState,
        decision: SignalDecision,
    ) -> StepOutcome:
        """Consume retained evidence and return a validated immutable transition."""


__all__ = [
    "ControllerContractError",
    "SignalController",
    "SignalEmission",
]
