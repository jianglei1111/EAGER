from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from eager.retrieval import RetrievalKernel
from eager.retrieval.contracts import RetrievalStepResult
from eager.schemas import (
    BudgetContract,
    ControllerState,
    InferencePayload,
    RetrievalState,
    SignalDecision,
    StepBudgetUsage,
    StepOutcome,
)
from eager.signals import SignalController


class RetrievalTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    decision: SignalDecision
    retrieval: RetrievalStepResult
    outcome: StepOutcome


class RetrievalFinalization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    rerank_pairs: int = Field(ge=0)


class RetrievalRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    payload: InferencePayload
    controller_type: str
    controller_state: ControllerState
    retrieval_state: RetrievalState
    steps: tuple[RetrievalTraceStep, ...]
    finalization: RetrievalFinalization | None = None


class RetrievalEngine:
    def __init__(self, kernel: RetrievalKernel) -> None:
        self.kernel = kernel

    def run(
        self,
        *,
        payload: InferencePayload,
        controller: SignalController,
        budget: BudgetContract,
    ) -> RetrievalRunResult:
        retrieval_state = RetrievalState.initial(budget)
        controller_state = controller.start(payload, budget)
        trace: list[RetrievalTraceStep] = []

        while controller_state.phase != "stopped":
            emission = controller.produce_signal(payload, retrieval_state, controller_state)
            controller_state = emission.state
            if emission.decision.action == "stop":
                break
            step = self.kernel.retrieve(
                payload=payload,
                signal=emission.decision,
                state=retrieval_state,
            )
            retrieval_state = retrieval_state.advance(
                signal=emission.decision,
                anchors=step.anchors,
                candidates=step.candidates,
                new_evidence=step.evidence,
                current_evidence=step.context_evidence,
                step_usage=step.usage,
            )
            outcome = controller.consume_evidence(
                payload,
                retrieval_state,
                controller_state,
                emission.decision,
            )
            if outcome.llm_calls or outcome.input_tokens or outcome.output_tokens:
                retrieval_state = retrieval_state.add_current_step_usage(
                    StepBudgetUsage(
                        llm_calls=outcome.llm_calls,
                        input_tokens=outcome.input_tokens,
                        output_tokens=outcome.output_tokens,
                    )
                )
                step = step.model_copy(update={"usage": retrieval_state.step_usage})
            controller_state = outcome.state
            trace.append(
                RetrievalTraceStep(
                    decision=emission.decision,
                    retrieval=step,
                    outcome=outcome,
                )
            )

        finalization_result = self.kernel.finalize(
            payload=payload,
            state=retrieval_state,
            candidate_steps=tuple(
                step.retrieval.coarse_candidates for step in trace
            ),
        )
        finalization = None
        if finalization_result is not None:
            retrieval_state = retrieval_state.finalize_evidence(
                tuple(candidate.evidence for candidate in finalization_result.candidates),
                rerank_pairs=finalization_result.rerank_pairs,
            )
            finalization = RetrievalFinalization(
                candidate_ids=tuple(
                    candidate.candidate_id for candidate in finalization_result.candidates
                ),
                evidence_ids=tuple(
                    candidate.evidence.evidence_id
                    for candidate in finalization_result.candidates
                ),
                rerank_pairs=finalization_result.rerank_pairs,
            )

        return RetrievalRunResult(
            payload=payload,
            controller_type=controller.controller_type,
            controller_state=controller_state,
            retrieval_state=retrieval_state,
            steps=tuple(trace),
            finalization=finalization,
        )


__all__ = [
    "RetrievalEngine",
    "RetrievalFinalization",
    "RetrievalRunResult",
    "RetrievalTraceStep",
]
