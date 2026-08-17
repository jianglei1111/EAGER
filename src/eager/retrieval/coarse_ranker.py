from __future__ import annotations

import re
import unicodedata

from eager.retrieval.anchors import generate_surface_keys, normalize_surface
from eager.retrieval.semantic import SemanticIndex
from eager.schemas import EvidenceCandidate, QueryAnchor, ScoreFeature


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def rank_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_PATTERN.findall(normalized))


def lexical_overlap_score(query_text: str, document_text: str) -> float:
    query_terms = set(rank_tokens(query_text))
    if not query_terms:
        return 0.0
    return len(query_terms & set(rank_tokens(document_text))) / len(query_terms)


def _feature(candidate: EvidenceCandidate, name: str) -> float:
    return next(
        (feature.value for feature in candidate.coarse_features if feature.name == name),
        0.0,
    )


def score_candidates(
    *,
    original_query: str,
    signal_query: str,
    anchors: tuple[QueryAnchor, ...],
    candidates: tuple[EvidenceCandidate, ...],
    semantic_index: SemanticIndex,
    original_query_weight: float,
    signal_query_weight: float,
) -> tuple[EvidenceCandidate, ...]:
    evidence_ids = tuple(candidate.evidence.evidence_id for candidate in candidates)
    total_weight = original_query_weight + signal_query_weight
    original_weight = original_query_weight / total_weight
    signal_weight = signal_query_weight / total_weight

    original_semantic = (
        semantic_index.score_many(original_query, evidence_ids)
        if original_weight > 0.0
        else {}
    )
    signal_semantic = (
        original_semantic
        if signal_weight > 0.0 and signal_query == original_query and original_semantic
        else semantic_index.score_many(signal_query, evidence_ids)
        if signal_weight > 0.0
        else {}
    )
    original_terms = set(rank_tokens(original_query))
    signal_terms = set(rank_tokens(signal_query))
    anchor_keys = tuple(generate_surface_keys(anchor.surface) for anchor in anchors)
    anchor_key_sets = tuple(set(keys) for keys in anchor_keys)

    scored: list[EvidenceCandidate] = []
    for candidate in candidates:
        evidence = candidate.evidence
        document = f"{evidence.title} {evidence.text}"
        document_terms = set(rank_tokens(document))
        original_lexical = (
            len(original_terms & document_terms) / len(original_terms)
            if original_terms
            else 0.0
        )
        signal_lexical = (
            len(signal_terms & document_terms) / len(signal_terms)
            if signal_terms
            else 0.0
        )
        lexical = (
            original_weight * original_lexical
            + signal_weight * signal_lexical
        )
        semantic = (
            original_weight * original_semantic.get(evidence.evidence_id, 0.0)
            + signal_weight * signal_semantic.get(evidence.evidence_id, 0.0)
        )
        title_keys = set(generate_surface_keys(evidence.title))
        title_contains = float(
            any(title_keys & keys for keys in anchor_key_sets)
        )
        normalized_document = normalize_surface(document)
        anchor_hits = sum(
            1
            for keys in anchor_keys
            if any(key and key in normalized_document for key in keys)
        )
        anchor_overlap = min(1.0, anchor_hits / max(1, len(anchors)))
        bridge_bonus = float(anchor_hits >= 2)

        similarity = 0.6 * lexical + 0.4 * semantic
        self_match = 0.85 * similarity + 0.15 * title_contains
        evidence_overlap = 0.7 * anchor_overlap + 0.3 * title_contains
        coarse_score = 0.55 * self_match + 0.30 * evidence_overlap + 0.15 * bridge_bonus
        scored.append(
            candidate.model_copy(
                update={
                    "coarse_features": (
                        ScoreFeature(name="lexical_score", value=lexical),
                        ScoreFeature(name="semantic_score", value=semantic),
                        ScoreFeature(name="title_contains", value=title_contains),
                        ScoreFeature(name="anchor_overlap", value=anchor_overlap),
                        ScoreFeature(name="bridge_bonus", value=bridge_bonus),
                    ),
                    "coarse_score": coarse_score,
                }
            )
        )

    scored.sort(key=lambda item: (-item.coarse_score, item.candidate_id))
    return tuple(
        candidate.model_copy(update={"coarse_rank": rank})
        for rank, candidate in enumerate(scored, start=1)
    )


__all__ = ["lexical_overlap_score", "rank_tokens", "score_candidates"]
