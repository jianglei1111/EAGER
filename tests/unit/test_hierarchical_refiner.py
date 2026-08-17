from __future__ import annotations

from eager.retrieval.hierarchical_refiner import refine_candidates
from eager.schemas import EvidenceCandidate, EvidenceUnit


class FixedReranker:
    def score_pairs(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(float(len(documents) - index) for index in range(len(documents)))


def _candidate(evidence_id: str, paragraph_id: str, score: float) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence=EvidenceUnit(
            evidence_id=evidence_id,
            paragraph_id=paragraph_id,
            sentence_id=evidence_id,
            title=paragraph_id,
            text=f"Evidence {evidence_id}",
        ),
        sources=("test",),
        coarse_score=score,
    )


def test_diversity_first_represents_each_selected_paragraph_before_filling() -> None:
    candidates = (
        _candidate("p1_s1", "p1", 4.0),
        _candidate("p1_s2", "p1", 3.0),
        _candidate("p2_s1", "p2", 2.0),
        _candidate("p3_s1", "p3", 1.0),
    )

    result = refine_candidates(
        query="question",
        candidates=candidates,
        reranker=FixedReranker(),
        hierarchical=True,
        paragraph_limit=3,
        rerank_limit=4,
        evidence_limit=3,
        diversity_first=True,
    )

    assert [candidate.evidence.paragraph_id for candidate in result.candidates] == [
        "p1",
        "p2",
        "p3",
    ]


def test_diversity_first_caps_representatives_then_keeps_ranked_remainder() -> None:
    candidates = (
        _candidate("p1_s1", "p1", 4.0),
        _candidate("p1_s2", "p1", 3.0),
        _candidate("p2_s1", "p2", 2.0),
        _candidate("p3_s1", "p3", 1.0),
    )

    result = refine_candidates(
        query="question",
        candidates=candidates,
        reranker=FixedReranker(),
        hierarchical=False,
        paragraph_limit=1,
        rerank_limit=4,
        evidence_limit=4,
        diversity_first=True,
        diversity_paragraph_limit=2,
    )

    assert [candidate.evidence.paragraph_id for candidate in result.candidates] == [
        "p1",
        "p2",
        "p1",
        "p3",
    ]
