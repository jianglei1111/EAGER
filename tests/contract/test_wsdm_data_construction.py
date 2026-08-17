from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eager.content_identity import make_paragraph_id
from eager.data.scaling import build_scaling_corpora
from eager.data.workflow import DataWorkflowError, _validate_annotation_support
from eager.indexing.corpus import CorpusRecord, CorpusSnapshot
from eager.indexing.structured import FORBIDDEN_STRUCTURED_FIELDS, build_structured_resources
from eager.indexing.surface_store import (
    build_sqlite_surface_store,
    build_surface_scope_manifest,
    validate_sqlite_surface_store,
)
from eager.retrieval import SqliteEntityIndex, extract_query_anchors
from eager.runtime.io import atomic_write_jsonl, sha256_file
from eager.schemas import EvaluationAnnotations


def _record(
    title: str,
    text: str,
    *,
    dataset: str = "hotpotqa",
    split: str = "fixture",
) -> CorpusRecord:
    sentences = (text,)
    return CorpusRecord(
        paragraph_id=make_paragraph_id(dataset=dataset, title=title, content=sentences),
        title=title,
        text=text,
        sentences=sentences,
        source_dataset=dataset,
        source_split=split,
    )


def _write_corpus(path: Path, records: list[CorpusRecord]) -> None:
    atomic_write_jsonl(path, (record.model_dump(mode="json") for record in records))


def _iter_keys(value: object):
    if isinstance(value, dict):
        yield from value
        for item in value.values():
            yield from _iter_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_keys(item)


def test_structured_resources_are_annotation_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spacy = pytest.importorskip("spacy")
    monkeypatch.setattr(spacy, "load", lambda _: spacy.blank("en"))
    corpus = CorpusSnapshot.from_records(
        [
            _record("Ada Lovelace", "Ada Lovelace worked on the Analytical Engine."),
            _record("!!!", "A punctuation-only title is still valid content."),
        ]
    )

    output = tmp_path / "structured"
    surface = tmp_path / "index" / "surface"
    summary = build_structured_resources(
        corpus,
        output,
        surface_index_dir=surface,
        spacy_model="fixture_blank_en",
    )

    assert summary["coreference_enabled"] is False
    assert summary["paragraph_count"] == 2
    for path in [*output.glob("*.json*"), *surface.glob("*.json*")]:
        if path.suffix == ".jsonl":
            payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        else:
            payloads = [json.loads(path.read_text(encoding="utf-8"))]
        for payload in payloads:
            assert not (set(_iter_keys(payload)) & FORBIDDEN_STRUCTURED_FIELDS)


def test_scaling_corpora_are_deterministic_nested_and_preserve_base(tmp_path: Path) -> None:
    base_records = [_record("Base", "alpha base evidence")]
    distractor_records = [
        _record(f"Distractor {index}", f"alpha distractor evidence {index}", dataset="2wiki")
        for index in range(1, 10)
    ]
    base_path = tmp_path / "base.jsonl"
    distractor_path = tmp_path / "distractor.jsonl"
    inference_path = tmp_path / "inference.jsonl"
    _write_corpus(base_path, base_records)
    _write_corpus(distractor_path, distractor_records)
    atomic_write_jsonl(
        inference_path,
        [{"dataset": "hotpotqa", "question": "Which alpha evidence is relevant?"}],
    )

    roots = [tmp_path / "run_a", tmp_path / "run_b"]
    manifests = [
        build_scaling_corpora(
            base_corpus=base_path,
            inference_path=inference_path,
            distractor_corpora=[distractor_path],
            output_root=root,
            repo_root=tmp_path,
        )
        for root in roots
    ]

    assert manifests[0] == manifests[1]
    assert sha256_file(roots[0] / "scaling_manifest.json") == sha256_file(
        roots[1] / "scaling_manifest.json"
    )
    base_id = base_records[0].paragraph_id
    previous: set[str] = set()
    for scale in (1, 2, 4, 8):
        rows = [
            json.loads(line)
            for line in (roots[0] / f"scale_{scale}x" / "corpus.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        current = {row["paragraph_id"] for row in rows}
        assert base_id in current
        assert not previous or previous < current
        assert len(current) == scale
        previous = current
        assert manifests[0]["scale_summaries"][str(scale)][
            "all_base_paragraphs_preserved"
        ]


def test_scaling_rejects_annotation_fields_in_inference(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    distractor_path = tmp_path / "distractor.jsonl"
    inference_path = tmp_path / "inference.jsonl"
    _write_corpus(base_path, [_record("Base", "base evidence")])
    _write_corpus(
        distractor_path,
        [_record(f"D{index}", f"distractor {index}", dataset="musique") for index in range(8)],
    )
    atomic_write_jsonl(
        inference_path,
        [{"dataset": "hotpotqa", "question": "Question?", "answer": "leaked"}],
    )

    with pytest.raises(ValueError, match="only dataset and question"):
        build_scaling_corpora(
            base_corpus=base_path,
            inference_path=inference_path,
            distractor_corpora=[distractor_path],
            output_root=tmp_path / "scales",
        )


def test_support_coverage_rejects_missing_gold_units() -> None:
    corpus = CorpusSnapshot.from_records([_record("Known", "known evidence")])
    annotation = EvaluationAnnotations(
        handle_id="qh_0123456789abcdef0123456789abcdef",
        answer="answer",
        supporting_paragraph_ids=["p_missing"],
    )

    with pytest.raises(DataWorkflowError, match="evaluation support is absent"):
        _validate_annotation_support([annotation], corpus)


def test_sqlite_surface_store_is_bound_to_nested_corpus_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spacy = pytest.importorskip("spacy")
    monkeypatch.setattr(spacy, "load", lambda _: spacy.blank("en"))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")
    base_record = _record("Base City", "Base City has a harbor.")
    base_record_other_split = CorpusRecord(
        **{
            **base_record.model_dump(mode="python"),
            "source_split": "other_fixture",
        }
    )
    extra_record = _record("Other City", "Other City has a station.", dataset="2wiki")
    base_path = tmp_path / "base.jsonl"
    larger_path = tmp_path / "larger.jsonl"
    _write_corpus(base_path, [base_record])
    _write_corpus(larger_path, [base_record_other_split, extra_record])
    store = tmp_path / "surface_store"

    summary = build_sqlite_surface_store(
        {"base": base_path, "larger": larger_path},
        store,
        spacy_model="fixture_blank_en",
        repo_root=tmp_path,
    )
    assert summary["unique_paragraph_count"] == 2
    assert validate_sqlite_surface_store(store) == summary

    surface = tmp_path / "indexes" / "base" / "surface"
    build_surface_scope_manifest(
        store_dir=store,
        scope="base",
        corpus_path=base_path,
        output_dir=surface,
        repo_root=tmp_path,
    )
    base_corpus = CorpusSnapshot.from_records([base_record])
    entity_index = SqliteEntityIndex.from_store_manifest(
        surface / "store_manifest.json",
        corpus=base_corpus,
    )
    anchors = extract_query_anchors("Where is Base City?", source="question", priority=0)
    paragraph_ids = {
        paragraph_id
        for match in entity_index.map_anchors(anchors)
        for paragraph_id in match.paragraph_ids
    }
    assert paragraph_ids == {base_record.paragraph_id}
    with ThreadPoolExecutor(max_workers=4) as executor:
        concurrent_results = tuple(
            executor.map(lambda _: entity_index.map_anchors(anchors), range(16))
        )
    assert all(result == concurrent_results[0] for result in concurrent_results)
    entity_index.close()

    with pytest.raises(ValueError, match="does not match the corpus"):
        SqliteEntityIndex.from_store_manifest(
            surface / "store_manifest.json",
            corpus=CorpusSnapshot.from_records([base_record, extra_record]),
        )
