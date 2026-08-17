from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eager.schemas import EvidenceCandidate, EvidenceUnit, QueryAnchor, StepBudgetUsage


RetrievalVariant = Literal[
    "full",
    "without_entity_expansion",
    "without_hierarchical_refinement",
]


class RetrievalKernelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    entity_expansion_enabled: bool = True
    evidence_derived_state_anchors_enabled: bool = True
    hierarchical_refinement_enabled: bool = True
    fine_reranking_enabled: bool = True
    global_fallback_enabled: bool = True
    diversity_first: bool = False
    diversity_paragraph_limit: int = Field(default=0, ge=0)
    candidate_limit: int = Field(default=20, ge=1)
    pre_score_candidate_limit: int | None = Field(default=None, ge=1)
    paragraph_limit: int = Field(default=5, ge=1)
    rerank_limit: int = Field(default=20, ge=1)
    final_rerank_limit: int = Field(default=0, ge=0)
    evidence_limit: int = Field(default=10, ge=1)
    # Evidence shown to the intermediate answerer for the current step. This
    # context is kept separate from the question-level retained-evidence cap.
    step_context_limit: int = Field(default=10, ge=1)
    original_query_weight: float = Field(default=0.4, ge=0.0)
    signal_query_weight: float = Field(default=0.6, ge=0.0)
    retained_anchor_scope: Literal["all", "latest_step"] = "all"
    retained_text_anchor_limit: int | None = Field(default=None, ge=0)
    candidate_union_enabled: bool = False
    candidate_union_semantic_limit: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_limits_and_weights(self) -> "RetrievalKernelConfig":
        if self.rerank_limit > self.candidate_limit:
            raise ValueError("rerank_limit cannot exceed candidate_limit")
        if (
            self.pre_score_candidate_limit is not None
            and self.pre_score_candidate_limit < self.candidate_limit
        ):
            raise ValueError("pre_score_candidate_limit cannot be below candidate_limit")
        if self.original_query_weight + self.signal_query_weight <= 0.0:
            raise ValueError("at least one query weight must be positive")
        return self


class RetrievalStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    anchors: tuple[QueryAnchor, ...]
    coarse_candidates: tuple[EvidenceCandidate, ...] = ()
    candidates: tuple[EvidenceCandidate, ...]
    evidence: tuple[EvidenceUnit, ...]
    context_evidence: tuple[EvidenceUnit, ...] = ()
    usage: StepBudgetUsage
    expanded_candidate_count: int = Field(ge=0)
    used_global_fallback: bool = False


def kernel_config_for_variant(variant: RetrievalVariant) -> RetrievalKernelConfig:
    if variant == "full":
        return RetrievalKernelConfig()
    if variant == "without_entity_expansion":
        return RetrievalKernelConfig(entity_expansion_enabled=False)
    if variant == "without_hierarchical_refinement":
        return RetrievalKernelConfig(hierarchical_refinement_enabled=False)
    raise ValueError(f"unknown retrieval variant: {variant}")


__all__ = [
    "RetrievalKernelConfig",
    "RetrievalStepResult",
    "RetrievalVariant",
    "kernel_config_for_variant",
]
