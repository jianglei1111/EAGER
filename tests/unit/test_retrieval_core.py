from __future__ import annotations

from pathlib import Path

import pytest

from eager.indexing import load_corpus_snapshot
from eager.retrieval import (
    EntityIndex,
    RetrievalKernelConfig,
    extract_query_anchors,
    generate_surface_keys,
    kernel_config_for_variant,
    lexical_overlap_score,
)
from eager.schemas import QueryAnchor


CORPUS = Path(__file__).parents[1] / "fixtures" / "indexing" / "corpus.jsonl"


def test_surface_normalization_and_entity_mapping_are_deterministic() -> None:
    snapshot = load_corpus_snapshot(CORPUS)
    index = EntityIndex.from_corpus(snapshot)
    anchors = extract_query_anchors(
        "Where is the company founded by Mira Sol headquartered?",
        source="question",
        priority=0,
    )

    assert [anchor.surface for anchor in anchors] == ["Mira Sol"]
    assert generate_surface_keys("Northstar Labs (company)") == (
        "northstar labs",
        "northstar labs (company)",
        "northstar labs company",
    )
    matches = index.map_anchors(anchors)
    assert len(matches) == 1
    assert matches[0].paragraph_ids == (snapshot.records[0].paragraph_id,)


def test_anchor_extraction_preserves_latin_unicode_names() -> None:
    anchors = extract_query_anchors(
        "Who directed Der König Von Kreuzberg?",
        source="question",
        priority=0,
    )

    assert [anchor.surface for anchor in anchors] == ["Der König Von Kreuzberg"]

def test_entity_mapping_uses_bounded_token_postings() -> None:
    snapshot = load_corpus_snapshot(CORPUS)
    index = EntityIndex.from_corpus(
        snapshot,
        max_tokens_per_anchor=4,
        max_postings_per_token=1,
        max_matches_per_anchor=1,
    )
    anchor = QueryAnchor.from_surface(
        "Background about Mira Sol",
        sources=["test"],
        priority=0,
    )

    match = index.map_anchor(anchor)

    assert match.paragraph_ids == (snapshot.records[0].paragraph_id,)
    assert all(len(postings) <= 1 for postings in index.token_to_paragraph_ids.values())


def test_legacy_lexical_overlap_formula_is_preserved() -> None:
    assert lexical_overlap_score(
        "director silent map",
        "The Silent Map was directed by Elian Voss.",
    ) == pytest.approx(2 / 3)
    assert lexical_overlap_score("", "anything") == 0.0


def test_retrieval_variant_changes_only_the_declared_component() -> None:
    full = kernel_config_for_variant("full")
    without_entity = kernel_config_for_variant("without_entity_expansion")
    without_hierarchy = kernel_config_for_variant("without_hierarchical_refinement")

    assert without_entity == full.model_copy(update={"entity_expansion_enabled": False})
    assert without_hierarchy == full.model_copy(
        update={"hierarchical_refinement_enabled": False}
    )


def test_kernel_config_allows_evidence_to_accumulate_across_steps() -> None:
    assert RetrievalKernelConfig(
        evidence_limit=30,
        rerank_limit=14,
        final_rerank_limit=0,
    ).evidence_limit == 30
    with pytest.raises(ValueError, match="rerank_limit cannot exceed candidate_limit"):
        RetrievalKernelConfig(rerank_limit=21, candidate_limit=20)
    with pytest.raises(
        ValueError, match="pre_score_candidate_limit cannot be below candidate_limit"
    ):
        RetrievalKernelConfig(candidate_limit=20, pre_score_candidate_limit=19)
