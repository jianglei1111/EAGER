from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

from eager.indexing import CorpusSnapshot
from eager.retrieval.anchors import (
    anchors_from_titles,
    extract_query_anchors,
    merge_anchors,
)
from eager.retrieval.candidates import build_candidates
from eager.retrieval.coarse_ranker import score_candidates
from eager.retrieval.contracts import RetrievalKernelConfig, RetrievalStepResult
from eager.retrieval.entity_mapping import EntityIndex, EntityMapper
from eager.retrieval.hierarchical_refiner import RefinementResult, refine_candidates
from eager.retrieval.reranker import LexicalReranker, PairReranker
from eager.retrieval.semantic import SemanticIndex
from eager.schemas import (
    EvidenceUnit,
    InferencePayload,
    RetrievalState,
    SignalDecision,
    StepBudgetUsage,
)


class RetrievalKernel:
    """Dataset-neutral candidate expansion, ranking, and evidence refinement."""

    def __init__(
        self,
        *,
        corpus: CorpusSnapshot,
        semantic_index: SemanticIndex,
        config: RetrievalKernelConfig,
        reranker: PairReranker | None = None,
        entity_index: EntityMapper | None = None,
        evidence_texts: Mapping[str, str] | None = None,
    ) -> None:
        if semantic_index.evidence_ids != {
            unit.evidence_id for unit in corpus.evidence_units
        }:
            raise ValueError("semantic index evidence IDs do not match the canonical corpus")
        representation = dict(evidence_texts or {})
        corpus_ids = {unit.evidence_id for unit in corpus.evidence_units}
        if evidence_texts is not None and set(representation) != corpus_ids:
            raise ValueError("evidence text IDs do not match the canonical corpus")
        if any(not isinstance(text, str) or not text.strip() for text in representation.values()):
            raise ValueError("evidence text representation contains an empty value")
        self.corpus = corpus
        self.semantic_index = semantic_index
        self.evidence_by_id = {
            unit.evidence_id: unit.model_copy(
                update={"text": representation.get(unit.evidence_id, unit.text)}
            )
            for unit in corpus.evidence_units
        }
        grouped_evidence = defaultdict(list)
        for unit in self.evidence_by_id.values():
            grouped_evidence[unit.paragraph_id].append(unit)
        self.evidence_by_paragraph = {
            paragraph_id: tuple(units)
            for paragraph_id, units in grouped_evidence.items()
        }
        self.config = config
        self.entity_index = entity_index or EntityIndex.from_corpus(corpus)
        self.reranker = reranker or LexicalReranker()

    def _retained_anchor_evidence(
        self,
        state: RetrievalState,
    ) -> tuple[EvidenceUnit, ...]:
        if self.config.retained_anchor_scope == "all":
            return state.retained_evidence
        retained_ids = {
            evidence.evidence_id for evidence in state.retained_evidence
        }
        return tuple(
            candidate.evidence
            for candidate in state.candidate_pool
            if candidate.evidence.evidence_id in retained_ids
        )

    def finalize(
        self,
        *,
        payload: InferencePayload,
        state: RetrievalState,
        candidate_steps: tuple[tuple[EvidenceCandidate, ...], ...],
    ) -> RefinementResult | None:
        if (
            not self.config.fine_reranking_enabled
            or self.config.final_rerank_limit == 0
        ):
            return None
        # Finalization protects retained evidence as a prefix. Once that prefix
        # fills the evidence budget, reranking cannot change the reader input.
        if len(state.retained_evidence) >= self.config.evidence_limit:
            return None
        unique_candidates: list[EvidenceCandidate] = []
        seen: set[str] = set()
        for candidate in (
            candidate for step in candidate_steps for candidate in step
        ):
            evidence_id = candidate.evidence.evidence_id
            if evidence_id not in seen:
                seen.add(evidence_id)
                unique_candidates.append(candidate)
        if not unique_candidates:
            return None
        rescored = score_candidates(
            original_query=payload.question,
            signal_query=payload.question,
            anchors=state.anchors,
            candidates=tuple(unique_candidates),
            semantic_index=self.semantic_index,
            original_query_weight=1.0,
            signal_query_weight=0.0,
        )
        remaining_pairs = max(
            0,
            state.budget.max_rerank_pairs_per_question
            - state.total_usage.rerank_pairs,
        )
        final_rerank_limit = min(
            self.config.final_rerank_limit,
            remaining_pairs,
            len(rescored),
        )
        if final_rerank_limit == 0:
            return None
        refined = refine_candidates(
            query=payload.question,
            candidates=rescored,
            reranker=self.reranker,
            hierarchical=False,
            paragraph_limit=1,
            rerank_limit=final_rerank_limit,
            evidence_limit=min(self.config.evidence_limit, final_rerank_limit),
            diversity_first=self.config.diversity_first,
            diversity_paragraph_limit=self.config.diversity_paragraph_limit,
        )
        # Keep evidence already selected by individual chain steps. The
        # legacy SC used the union of step-local top evidence; a pure global
        # rerank can discard a remote bridge fact that is weakly related to the
        # original question but essential to one dependency branch.
        by_evidence_id = {
            candidate.evidence.evidence_id: candidate for candidate in rescored
        }
        protected = [
            by_evidence_id[evidence.evidence_id]
            for evidence in state.retained_evidence
            if evidence.evidence_id in by_evidence_id
        ]
        merged: list[EvidenceCandidate] = []
        seen_ids: set[str] = set()
        for candidate in (*protected, *refined.candidates):
            if candidate.evidence.evidence_id in seen_ids:
                continue
            seen_ids.add(candidate.evidence.evidence_id)
            merged.append(candidate)
        limit = min(self.config.evidence_limit, final_rerank_limit, len(merged))
        selected = tuple(
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(merged[:limit], start=1)
        )
        return RefinementResult(
            candidates=selected,
            parent_units_selected=refined.parent_units_selected,
            rerank_pairs=refined.rerank_pairs,
        )

    def retrieve(
        self,
        *,
        payload: InferencePayload,
        signal: SignalDecision,
        state: RetrievalState,
    ) -> RetrievalStepResult:
        signal_query = signal.signal_text or payload.question
        question_anchors = extract_query_anchors(
            payload.question,
            source="question",
            priority=0,
        )
        signal_anchors = extract_query_anchors(
            signal_query,
            source="signal",
            priority=1,
        )
        retained_anchor_evidence = (
            self._retained_anchor_evidence(state)
            if self.config.evidence_derived_state_anchors_enabled
            else ()
        )
        retained_anchors = anchors_from_titles(
            evidence.title for evidence in retained_anchor_evidence
        )
        retained_text_anchors = merge_anchors(
            *(
                extract_query_anchors(
                    evidence.text,
                    source="retained_evidence",
                    priority=2,
                    max_anchors=8,
                )
                for evidence in retained_anchor_evidence
            )
        )
        if self.config.retained_text_anchor_limit is not None:
            retained_text_anchors = retained_text_anchors[
                : self.config.retained_text_anchor_limit
            ]
        anchors = merge_anchors(
            question_anchors,
            signal_anchors,
            retained_anchors,
            retained_text_anchors,
        )
        candidates, used_global_fallback = build_candidates(
            evidence_by_id=self.evidence_by_id,
            evidence_by_paragraph=self.evidence_by_paragraph,
            entity_index=self.entity_index,
            semantic_index=self.semantic_index,
            anchors=anchors,
            fallback_query=signal_query,
            fallback_limit=self.config.candidate_limit,
            entity_expansion_enabled=self.config.entity_expansion_enabled,
            global_fallback_enabled=self.config.global_fallback_enabled,
            candidate_union_enabled=self.config.candidate_union_enabled,
            candidate_union_semantic_limit=self.config.candidate_union_semantic_limit,
            pre_score_candidate_limit=self.config.pre_score_candidate_limit,
        )
        expanded_candidate_count = len(candidates)
        ranked = score_candidates(
            original_query=payload.question,
            signal_query=signal_query,
            anchors=anchors,
            candidates=candidates,
            semantic_index=self.semantic_index,
            original_query_weight=self.config.original_query_weight,
            signal_query_weight=self.config.signal_query_weight,
        )

        remaining_candidates = max(
            0,
            state.budget.max_candidates_per_question
            - state.total_usage.candidates_considered,
        )
        candidate_limit = min(
            self.config.candidate_limit,
            state.budget.max_candidates_per_step,
            remaining_candidates,
        )
        retained_evidence_ids = {
            evidence.evidence_id for evidence in state.retained_evidence
        }
        novel_ranked = tuple(
            candidate
            for candidate in ranked
            if candidate.evidence.evidence_id not in retained_evidence_ids
        )
        if self.config.candidate_union_enabled:
            semantic_quota = min(
                candidate_limit,
                self.config.candidate_union_semantic_limit,
            )
            entity_quota = candidate_limit - semantic_quota
            semantic_lane = [
                candidate
                for candidate in novel_ranked
                if "semantic" in candidate.sources
            ]
            entity_lane = [
                candidate
                for candidate in novel_ranked
                if "entity" in candidate.sources
            ]
            selected: list[EvidenceCandidate] = []
            selected_ids: set[str] = set()
            for candidate in (*semantic_lane[:semantic_quota], *entity_lane[:entity_quota]):
                if candidate.evidence.evidence_id in selected_ids:
                    continue
                selected_ids.add(candidate.evidence.evidence_id)
                selected.append(candidate)
            for candidate in novel_ranked:
                if len(selected) >= candidate_limit:
                    break
                if candidate.evidence.evidence_id in selected_ids:
                    continue
                selected_ids.add(candidate.evidence.evidence_id)
                selected.append(candidate)
            novel_ranked = tuple(
                sorted(selected, key=lambda item: (item.coarse_rank or 0, item.candidate_id))
            )
        novel_ranked = novel_ranked[:candidate_limit]
        ranked = novel_ranked
        remaining_parents = max(
            0,
            state.budget.max_parent_units_per_question
            - state.total_usage.parent_units_selected,
        )
        paragraph_limit = min(
            self.config.paragraph_limit,
            state.budget.max_parent_units_per_step,
            remaining_parents,
        )
        remaining_pairs = max(
            0,
            state.budget.max_rerank_pairs_per_question - state.total_usage.rerank_pairs,
        )
        rerank_limit = min(
            self.config.rerank_limit,
            state.budget.max_rerank_pairs_per_step,
            remaining_pairs,
        )
        remaining_evidence = max(
            0,
            state.budget.max_retained_evidence - len(state.retained_evidence),
        )
        if not self.config.fine_reranking_enabled:
            selected = novel_ranked[: min(self.config.evidence_limit, remaining_evidence)]
            refined_candidates = tuple(
                candidate.model_copy(
                    update={"fine_score": candidate.coarse_score, "rank": rank}
                )
                for rank, candidate in enumerate(selected, start=1)
            )
            context_evidence = tuple(
                candidate.evidence
                for candidate in novel_ranked[: self.config.step_context_limit]
            )
            parent_units_selected = 0
            rerank_pairs = 0
        elif paragraph_limit == 0 or rerank_limit == 0 or not novel_ranked:
            refined_candidates = ()
            context_evidence = ()
            parent_units_selected = 0
            rerank_pairs = 0
        else:
            retained_evidence_limit = min(
                self.config.evidence_limit,
                max(
                    1,
                    math.ceil(
                        remaining_evidence
                        / max(1, state.budget.max_iterations - state.iteration)
                    ),
                ),
                remaining_evidence,
            )
            # Keep a larger current-step context for the intermediate answerer
            # even when the question-level evidence cap is nearly exhausted.
            refinement_evidence_limit = min(
                self.config.rerank_limit,
                max(retained_evidence_limit, self.config.step_context_limit),
            )
            refined = refine_candidates(
                query=signal_query,
                candidates=novel_ranked,
                reranker=self.reranker,
                hierarchical=self.config.hierarchical_refinement_enabled,
                paragraph_limit=paragraph_limit,
                rerank_limit=rerank_limit,
                evidence_limit=refinement_evidence_limit,
                diversity_first=self.config.diversity_first,
                diversity_paragraph_limit=self.config.diversity_paragraph_limit,
            )
            refined_candidates = refined.candidates[:retained_evidence_limit]
            context_evidence = tuple(
                candidate.evidence
                for candidate in refined.candidates[: self.config.step_context_limit]
            )
            parent_units_selected = refined.parent_units_selected
            rerank_pairs = refined.rerank_pairs
        usage = StepBudgetUsage(
            candidates_considered=len(ranked),
            parent_units_selected=parent_units_selected,
            rerank_pairs=rerank_pairs,
            llm_calls=signal.llm_calls,
            input_tokens=signal.input_tokens,
            output_tokens=signal.output_tokens,
        )
        return RetrievalStepResult(
            anchors=anchors,
            coarse_candidates=ranked,
            candidates=refined_candidates,
            evidence=tuple(candidate.evidence for candidate in refined_candidates),
            context_evidence=context_evidence,
            usage=usage,
            expanded_candidate_count=expanded_candidate_count,
            used_global_fallback=used_global_fallback,
        )


__all__ = ["RetrievalKernel"]
