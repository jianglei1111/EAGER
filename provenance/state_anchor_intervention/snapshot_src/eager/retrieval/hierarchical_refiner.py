from __future__ import annotations

from dataclasses import dataclass

from eager.retrieval.reranker import PairReranker
from eager.schemas import EvidenceCandidate


@dataclass(frozen=True, slots=True)
class RefinementResult:
    candidates: tuple[EvidenceCandidate, ...]
    parent_units_selected: int
    rerank_pairs: int


def _diversity_first_order(
    candidates: tuple[EvidenceCandidate, ...] | list[EvidenceCandidate],
    *,
    paragraph_limit: int,
) -> list[EvidenceCandidate]:
    representatives: list[EvidenceCandidate] = []
    remainder: list[EvidenceCandidate] = []
    seen_paragraphs: set[str] = set()
    for candidate in candidates:
        paragraph_id = candidate.evidence.paragraph_id
        if paragraph_id in seen_paragraphs or (
            paragraph_limit > 0 and len(seen_paragraphs) >= paragraph_limit
        ):
            remainder.append(candidate)
        else:
            seen_paragraphs.add(paragraph_id)
            representatives.append(candidate)
    return [*representatives, *remainder]


def _title_bonus(candidate: EvidenceCandidate) -> float:
    return next(
        (
            feature.value
            for feature in candidate.coarse_features
            if feature.name == "title_contains"
        ),
        0.0,
    )


def refine_candidates(
    *,
    query: str,
    candidates: tuple[EvidenceCandidate, ...],
    reranker: PairReranker,
    hierarchical: bool,
    paragraph_limit: int,
    rerank_limit: int,
    evidence_limit: int,
    diversity_first: bool = False,
    diversity_paragraph_limit: int = 0,
) -> RefinementResult:
    if not candidates:
        return RefinementResult(candidates=(), parent_units_selected=0, rerank_pairs=0)

    if hierarchical:
        paragraph_scores: dict[str, float] = {}
        for candidate in candidates:
            paragraph_id = candidate.evidence.paragraph_id
            paragraph_scores[paragraph_id] = max(
                paragraph_scores.get(paragraph_id, float("-inf")),
                candidate.coarse_score,
            )
        ranked_paragraphs = sorted(
            paragraph_scores,
            key=lambda paragraph_id: (-paragraph_scores[paragraph_id], paragraph_id),
        )[:paragraph_limit]
        paragraph_ranks = {
            paragraph_id: rank for rank, paragraph_id in enumerate(ranked_paragraphs, start=1)
        }
        eligible = tuple(
            candidate.model_copy(
                update={"paragraph_rank": paragraph_ranks[candidate.evidence.paragraph_id]}
            )
            for candidate in candidates
            if candidate.evidence.paragraph_id in paragraph_ranks
        )
    else:
        eligible = candidates

    rerank_order = (
        _diversity_first_order(
            eligible,
            paragraph_limit=diversity_paragraph_limit,
        )
        if diversity_first
        else list(eligible)
    )
    rerank_pool = tuple(rerank_order[:rerank_limit])
    documents = tuple(
        f"{candidate.evidence.title} [SEP] {candidate.evidence.text}"
        for candidate in rerank_pool
    )
    scores = tuple(float(score) for score in reranker.score_pairs(query, documents))
    if len(scores) != len(rerank_pool):
        raise ValueError("reranker returned a score count different from the document count")
    reranked = [
        candidate.model_copy(update={"fine_score": score + 0.1 * _title_bonus(candidate)})
        for candidate, score in zip(rerank_pool, scores)
    ]
    reranked.sort(
        key=lambda item: (
            -(item.fine_score if item.fine_score is not None else float("-inf")),
            -item.coarse_score,
            item.candidate_id,
        )
    )
    if diversity_first:
        reranked = _diversity_first_order(
            reranked,
            paragraph_limit=diversity_paragraph_limit,
        )
    final = tuple(
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(reranked[:evidence_limit], start=1)
    )
    selected_paragraphs = (
        {candidate.evidence.paragraph_id for candidate in eligible} if hierarchical else set()
    )
    return RefinementResult(
        candidates=final,
        parent_units_selected=len(selected_paragraphs),
        rerank_pairs=len(rerank_pool),
    )


__all__ = ["RefinementResult", "refine_candidates"]
