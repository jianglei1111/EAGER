from __future__ import annotations

import json
from pathlib import Path

import pytest

from eager import cli
from eager.indexing import build_tiny_index, load_corpus_snapshot
from eager.pipeline import RetrievalEngine
from eager.retrieval import (
    RetrievalKernel,
    RetrievalKernelConfig,
    TinySemanticIndex,
)
from eager.retrieval.candidates import build_candidates
from eager.retrieval.entity_mapping import EntityMatch
from eager.schemas import (
    BudgetContract,
    EvidenceUnit,
    FORBIDDEN_INFERENCE_FIELDS,
    InferencePayload,
    QueryAnchor,
)
from eager.signals import NoSignalController, RetrievalQuestionController


CORPUS_FIXTURE = Path(__file__).parents[1] / "fixtures" / "indexing" / "corpus.jsonl"
QUESTION = "Where is the company founded by Mira Sol headquartered?"


def _budget(*, max_iterations: int = 2) -> BudgetContract:
    return BudgetContract(
        max_iterations=max_iterations,
        max_candidates_per_step=10,
        max_candidates_per_question=20,
        max_parent_units_per_step=5,
        max_parent_units_per_question=10,
        max_rerank_pairs_per_step=10,
        max_rerank_pairs_per_question=20,
        max_retained_evidence=5,
        max_llm_calls_per_step=0,
        max_llm_calls_per_question=0,
        max_input_tokens_per_question=0,
        max_output_tokens_per_question=0,
    )


def _engine(
    tmp_path: Path,
    *,
    evidence_texts: dict[str, str] | None = None,
    **config: object,
) -> RetrievalEngine:
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS_FIXTURE.read_bytes())
    corpus = load_corpus_snapshot(corpus_path)
    index_dir = tmp_path / "index"
    build_tiny_index(corpus, index_dir, dimensions=32)
    semantic = TinySemanticIndex.load(index_dir, corpus=corpus)
    defaults = {
        "candidate_limit": 10,
        "paragraph_limit": 5,
        "rerank_limit": 10,
        "evidence_limit": 5,
    }
    return RetrievalEngine(
        RetrievalKernel(
            corpus=corpus,
            semantic_index=semantic,
            config=RetrievalKernelConfig(**{**defaults, **config}),
            evidence_texts=evidence_texts,
        )
    )


def test_kernel_uses_alternate_text_without_changing_evidence_ids(tmp_path: Path) -> None:
    corpus = load_corpus_snapshot(CORPUS_FIXTURE)
    evidence_texts = {
        unit.evidence_id: f"resolved: {unit.text}"
        for unit in corpus.evidence_units
    }

    engine = _engine(tmp_path, evidence_texts=evidence_texts)

    assert set(engine.kernel.evidence_by_id) == set(evidence_texts)
    assert {
        evidence_id: unit.text
        for evidence_id, unit in engine.kernel.evidence_by_id.items()
    } == evidence_texts


def test_retrieval_question_engine_exposes_bridge_anchor_on_second_hop(
    tmp_path: Path,
) -> None:
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=RetrievalQuestionController(max_steps=2),
        budget=_budget(),
    )

    assert result.controller_state.phase == "stopped"
    assert result.retrieval_state.iteration == 2
    assert [step.retrieval.expanded_candidate_count for step in result.steps] == [1, 3]
    assert [
        [candidate.candidate_id for candidate in step.retrieval.candidates]
        for step in result.steps
    ] == [
        ["hp_par_1152dcc0582f9aca91a27043217d4dfe_s0"],
        [
            "hp_par_dbc91f9b737e499293ed8a58c84c6872_s1",
            "hp_par_dbc91f9b737e499293ed8a58c84c6872_s0",
        ],
    ]
    assert result.steps[0].retrieval.candidates[0].coarse_score == pytest.approx(
        0.5825531389897809
    )
    assert "Northstar Labs" in [anchor.surface for anchor in result.steps[1].retrieval.anchors]
    retained_titles = {item.title for item in result.retrieval_state.retained_evidence}
    assert retained_titles == {"Mira Sol", "Northstar Labs"}
    assert len(result.retrieval_state.retained_evidence) == 3
    assert not (
        {
            item.evidence_id
            for item in result.steps[0].retrieval.evidence
        }
        & {
            item.evidence_id
            for item in result.steps[1].retrieval.evidence
        }
    )
    assert result.retrieval_state.total_usage.llm_calls == 0

    serialized_payload = result.payload.model_dump(mode="json")
    assert serialized_payload == {"dataset": "2wiki", "question": QUESTION}
    assert not (set(serialized_payload) & FORBIDDEN_INFERENCE_FIELDS)


def test_candidate_union_keeps_entity_and_semantic_lanes(tmp_path: Path) -> None:
    result = _engine(
        tmp_path,
        candidate_union_enabled=True,
        candidate_union_semantic_limit=3,
    ).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=NoSignalController(max_steps=1),
        budget=_budget(max_iterations=1),
    )

    candidates = result.steps[0].retrieval.coarse_candidates
    assert candidates
    assert any("entity" in candidate.sources for candidate in candidates)
    assert any("semantic" in candidate.sources for candidate in candidates)
    assert result.steps[0].retrieval.expanded_candidate_count >= len(candidates)


def test_pre_score_budget_stops_entity_expansion_before_coarse_scoring() -> None:
    def units(paragraph_id: str, count: int) -> tuple[EvidenceUnit, ...]:
        return tuple(
            EvidenceUnit(
                evidence_id=f"{paragraph_id}_s{index}",
                paragraph_id=paragraph_id,
                sentence_id=f"{paragraph_id}_s{index}",
                title=paragraph_id,
                text=f"{paragraph_id} sentence {index}",
            )
            for index in range(count)
        )

    evidence_by_paragraph = {
        "p1": units("p1", 3),
        "p2": units("p2", 3),
        "p3": units("p3", 3),
        "p4": units("p4", 1),
    }
    evidence_by_id = {
        unit.evidence_id: unit
        for paragraph_units in evidence_by_paragraph.values()
        for unit in paragraph_units
    }

    class EntityMapperStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def map_anchor(self, anchor: QueryAnchor) -> EntityMatch:
            self.calls.append(anchor.surface)
            paragraph_ids = ("p1", "p2") if "Current" in anchor.surface else ("p3",)
            return EntityMatch(anchor_id=anchor.anchor_id, paragraph_ids=paragraph_ids)

    class SemanticIndexStub:
        def top_k(self, query: str, limit: int) -> tuple[tuple[str, float], ...]:
            assert query == "current relation"
            assert limit == 1
            return (("p4_s0", 1.0),)

    entity_index = EntityMapperStub()
    candidates, fallback = build_candidates(
        evidence_by_id=evidence_by_id,
        evidence_by_paragraph=evidence_by_paragraph,
        entity_index=entity_index,
        semantic_index=SemanticIndexStub(),
        anchors=(
            QueryAnchor.from_surface("Original Entity", sources=["question"], priority=0),
            QueryAnchor.from_surface("Current Entity", sources=["signal"], priority=1),
        ),
        fallback_query="current relation",
        fallback_limit=5,
        entity_expansion_enabled=True,
        global_fallback_enabled=True,
        candidate_union_enabled=True,
        candidate_union_semantic_limit=1,
        pre_score_candidate_limit=5,
    )

    assert fallback is False
    assert entity_index.calls == ["Current Entity"]
    assert len(candidates) == 5
    assert sum("entity" in candidate.sources for candidate in candidates) == 4
    assert sum("semantic" in candidate.sources for candidate in candidates) == 1


def test_cross_step_candidate_limit_refills_after_excluding_retained_evidence(
    tmp_path: Path,
) -> None:
    result = _engine(
        tmp_path,
        candidate_limit=1,
        paragraph_limit=1,
        rerank_limit=1,
        evidence_limit=1,
    ).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=NoSignalController(max_steps=2),
        budget=_budget(),
    )

    first_ids = {
        candidate.evidence.evidence_id
        for candidate in result.steps[0].retrieval.coarse_candidates
    }
    second_ids = {
        candidate.evidence.evidence_id
        for candidate in result.steps[1].retrieval.coarse_candidates
    }
    assert len(first_ids) == 1
    assert len(second_ids) == 1
    assert not (first_ids & second_ids)


def test_latest_step_anchor_scope_excludes_older_retained_evidence(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, retained_anchor_scope="latest_step")
    result = engine.run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=RetrievalQuestionController(max_steps=2),
        budget=_budget(),
    )

    anchor_evidence = engine.kernel._retained_anchor_evidence(
        result.retrieval_state
    )
    latest_candidate_ids = {
        candidate.evidence.evidence_id
        for candidate in result.retrieval_state.candidate_pool
    }
    assert {evidence.evidence_id for evidence in anchor_evidence} == (
        latest_candidate_ids
        & {
            evidence.evidence_id
            for evidence in result.retrieval_state.retained_evidence
        }
    )
    assert not (
        {evidence.evidence_id for evidence in anchor_evidence}
        & {
            evidence.evidence_id
            for evidence in result.steps[0].retrieval.evidence
        }
    )


def test_one_engine_covers_component_and_signal_controls(tmp_path: Path) -> None:
    payload = InferencePayload(dataset="2wiki", question=QUESTION)
    full = _engine(tmp_path / "full").run(
        payload=payload,
        controller=RetrievalQuestionController(max_steps=1),
        budget=_budget(max_iterations=1),
    )
    no_entity = _engine(
        tmp_path / "no-entity",
        entity_expansion_enabled=False,
    ).run(
        payload=payload,
        controller=RetrievalQuestionController(max_steps=1),
        budget=_budget(max_iterations=1),
    )
    no_hierarchy = _engine(
        tmp_path / "no-hierarchy",
        hierarchical_refinement_enabled=False,
    ).run(
        payload=payload,
        controller=RetrievalQuestionController(max_steps=1),
        budget=_budget(max_iterations=1),
    )
    dense_only = _engine(
        tmp_path / "dense-only",
        entity_expansion_enabled=False,
        hierarchical_refinement_enabled=False,
        fine_reranking_enabled=False,
    ).run(
        payload=payload,
        controller=RetrievalQuestionController(max_steps=1),
        budget=_budget(max_iterations=1),
    )
    no_signal = _engine(tmp_path / "no-signal").run(
        payload=payload,
        controller=NoSignalController(max_steps=1),
        budget=_budget(max_iterations=1),
    )

    assert full.steps[0].retrieval.expanded_candidate_count == 1
    assert no_entity.steps[0].retrieval.expanded_candidate_count == 3
    assert no_hierarchy.steps[0].retrieval.usage.parent_units_selected == 0
    assert dense_only.steps[0].retrieval.usage.rerank_pairs == 0
    assert all(
        candidate.fine_score == candidate.coarse_score
        for candidate in dense_only.steps[0].retrieval.candidates
    )
    assert no_signal.controller_type == "no_signal"
    assert full.controller_type == "retrieval_question"


def test_final_rerank_selects_from_the_cross_step_candidate_union(
    tmp_path: Path,
) -> None:
    result = _engine(tmp_path, final_rerank_limit=5).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=RetrievalQuestionController(max_steps=2),
        budget=_budget(),
    )

    assert result.finalization is not None
    assert result.finalization.rerank_pairs == 3
    assert set(result.finalization.evidence_ids) == {
        candidate.evidence.evidence_id
        for step in result.steps
        for candidate in step.retrieval.coarse_candidates
    }
    assert result.retrieval_state.total_usage.rerank_pairs == (
        sum(step.retrieval.usage.rerank_pairs for step in result.steps)
        + result.finalization.rerank_pairs
    )


def test_final_rerank_is_skipped_when_retained_evidence_is_full(
    tmp_path: Path,
) -> None:
    result = _engine(
        tmp_path,
        evidence_limit=3,
        final_rerank_limit=5,
    ).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=RetrievalQuestionController(max_steps=2),
        budget=_budget(),
    )

    assert len(result.retrieval_state.retained_evidence) == 3
    assert result.finalization is None
    assert result.retrieval_state.total_usage.rerank_pairs == sum(
        step.retrieval.usage.rerank_pairs for step in result.steps
    )


def test_final_rerank_pair_limit_does_not_truncate_retained_evidence(
    tmp_path: Path,
) -> None:
    result = _engine(
        tmp_path,
        evidence_limit=5,
        final_rerank_limit=1,
    ).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=RetrievalQuestionController(max_steps=2),
        budget=_budget(),
    )

    retained_before_finalization = {
        evidence.evidence_id
        for step in result.steps
        for evidence in step.retrieval.evidence
    }
    retained_after_finalization = {
        evidence.evidence_id for evidence in result.retrieval_state.retained_evidence
    }

    assert result.finalization is not None
    assert result.finalization.rerank_pairs == 1
    assert retained_before_finalization <= retained_after_finalization


def test_final_rerank_protects_retained_evidence_outside_candidate_union(
    tmp_path: Path,
) -> None:
    engine = _engine(
        tmp_path,
        evidence_limit=5,
        final_rerank_limit=5,
    )
    result = engine.run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=RetrievalQuestionController(max_steps=2),
        budget=_budget(),
    )
    retained_ids = tuple(
        evidence.evidence_id for evidence in result.retrieval_state.retained_evidence
    )
    excluded_id = retained_ids[0]
    candidate_steps = tuple(
        tuple(
            candidate
            for candidate in step.retrieval.coarse_candidates
            if candidate.evidence.evidence_id != excluded_id
        )
        for step in result.steps
    )

    assert all(
        candidate.evidence.evidence_id != excluded_id
        for step in candidate_steps
        for candidate in step
    )
    finalization = engine.kernel.finalize(
        payload=result.payload,
        state=result.retrieval_state,
        candidate_steps=candidate_steps,
    )

    assert finalization is not None
    assert tuple(
        candidate.evidence.evidence_id
        for candidate in finalization.candidates[: len(retained_ids)]
    ) == retained_ids


def test_retrieve_cli_runs_without_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(CORPUS_FIXTURE.read_bytes())
    build_tiny_index(corpus_path, tmp_path / "index", dimensions=32)
    monkeypatch.chdir(tmp_path)

    assert cli.main(
        [
            "retrieve",
            "--dataset",
            "2wiki",
            "--question",
            QUESTION,
            "--corpus",
            "corpus.jsonl",
            "--index",
            "index",
            "--max-iterations",
            "1",
            "--candidate-limit",
            "3",
            "--paragraph-limit",
            "2",
            "--rerank-limit",
            "3",
            "--evidence-limit",
            "2",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    output = json.loads(captured.out)
    assert output["ok"] is True
    assert output["result"]["payload"] == {"dataset": "2wiki", "question": QUESTION}
