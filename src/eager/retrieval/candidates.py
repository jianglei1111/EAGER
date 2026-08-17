from __future__ import annotations

from collections.abc import Mapping

from eager.retrieval.entity_mapping import EntityMapper
from eager.retrieval.semantic import SemanticIndex
from eager.schemas import EvidenceCandidate, EvidenceUnit, QueryAnchor


def build_candidates(
    *,
    evidence_by_id: Mapping[str, EvidenceUnit],
    evidence_by_paragraph: Mapping[str, tuple[EvidenceUnit, ...]],
    entity_index: EntityMapper,
    semantic_index: SemanticIndex,
    anchors: tuple[QueryAnchor, ...],
    fallback_query: str,
    fallback_limit: int,
    entity_expansion_enabled: bool,
    global_fallback_enabled: bool,
    candidate_union_enabled: bool = False,
    candidate_union_semantic_limit: int | None = None,
    pre_score_candidate_limit: int | None = None,
) -> tuple[tuple[EvidenceCandidate, ...], bool]:
    semantic_limit = max(
        1,
        min(
            fallback_limit,
            candidate_union_semantic_limit or fallback_limit,
        ),
    )
    semantic_ids = (
        tuple(evidence_id for evidence_id, _ in semantic_index.top_k(fallback_query, semantic_limit))
        if candidate_union_enabled
        else ()
    )

    paragraph_anchor_ids: dict[str, list[str]] = {}
    entity_units: dict[str, EvidenceUnit] = {}
    if entity_expansion_enabled:
        if pre_score_candidate_limit is None:
            for match in entity_index.map_anchors(anchors):
                for paragraph_id in match.paragraph_ids:
                    paragraph_anchor_ids.setdefault(paragraph_id, []).append(match.anchor_id)
            entity_units = {
                unit.evidence_id: unit
                for paragraph_id in sorted(paragraph_anchor_ids)
                for unit in evidence_by_paragraph.get(paragraph_id, ())
            }
        else:
            entity_limit = max(0, pre_score_candidate_limit - len(semantic_ids))

            def anchor_rank(anchor: QueryAnchor) -> tuple[int, int, str]:
                sources = set(anchor.sources)
                source_rank = (
                    0
                    if "signal" in sources
                    else 1
                    if "question" in sources
                    else 2
                    if "retained_title" in sources
                    else 3
                )
                return source_rank, anchor.priority, anchor.normalized_surface

            for anchor in sorted(anchors, key=anchor_rank):
                match = entity_index.map_anchor(anchor)
                for paragraph_id in match.paragraph_ids:
                    units = evidence_by_paragraph.get(paragraph_id, ())
                    selected_from_paragraph = False
                    for unit in units:
                        if unit.evidence_id in entity_units:
                            selected_from_paragraph = True
                            continue
                        if len(entity_units) >= entity_limit:
                            break
                        entity_units[unit.evidence_id] = unit
                        selected_from_paragraph = True
                    if selected_from_paragraph:
                        paragraph_anchor_ids.setdefault(paragraph_id, []).append(
                            match.anchor_id
                        )
                    if len(entity_units) >= entity_limit:
                        break
                if len(entity_units) >= entity_limit:
                    break

    used_global_fallback = False

    if entity_units and (not candidate_union_enabled or not semantic_ids):
        candidates = tuple(
            EvidenceCandidate(
                evidence=unit,
                sources=["entity"],
                anchor_ids=paragraph_anchor_ids.get(unit.paragraph_id, ()),
                coarse_score=0.0,
            )
            for unit in entity_units.values()
        )
        return candidates, False

    if not entity_units and not semantic_ids:
        if entity_expansion_enabled and not global_fallback_enabled:
            return (), False
        used_global_fallback = entity_expansion_enabled
        semantic_ids = tuple(
            evidence_id
            for evidence_id, _ in semantic_index.top_k(fallback_query, fallback_limit)
        )

    merged: dict[str, dict[str, object]] = {}
    for evidence_id, unit in entity_units.items():
        merged[evidence_id] = {
            "evidence": unit,
            "sources": {"entity"},
            "anchor_ids": set(paragraph_anchor_ids.get(unit.paragraph_id, ())),
        }
    for evidence_id in semantic_ids:
        unit = evidence_by_id[evidence_id]
        record = merged.setdefault(
            evidence_id,
            {"evidence": unit, "sources": set(), "anchor_ids": set()},
        )
        record["sources"].add("semantic")

    candidates = tuple(
        EvidenceCandidate(
            evidence=record["evidence"],
            sources=tuple(sorted(record["sources"])),
            anchor_ids=tuple(sorted(record["anchor_ids"])),
            coarse_score=0.0,
        )
        for evidence_id, record in sorted(merged.items())
    )
    return candidates, used_global_fallback


__all__ = ["build_candidates"]
