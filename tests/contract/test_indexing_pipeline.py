from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from eager.indexing import (
    CorpusValidationError,
    ModelArtifactPin,
    ModelPin,
    IndexValidationError,
    TINY_MODEL_PIN,
    artifact_manifest_sha256,
    build_dense_index,
    build_tiny_index,
    load_corpus_snapshot,
    model_pin_sha256,
    validate_corpus,
    validate_dense_index,
    validate_index,
)
from eager.retrieval import DenseSemanticIndex


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "indexing" / "corpus.jsonl"


class _FixtureEncoder:
    def get_sentence_embedding_dimension(self) -> int:
        return 8

    def encode(self, sentences, **_kwargs):
        rows = []
        for sentence in sentences:
            digest = hashlib.sha256(sentence.encode("utf-8")).digest()
            vector = np.asarray([byte - 127.5 for byte in digest[:8]], dtype=np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32)


def _fixture_model(root: Path) -> tuple[ModelPin, Path]:
    snapshot = root / "model"
    snapshot.mkdir(parents=True)
    weights = b"fixture-model-weights"
    (snapshot / "weights.bin").write_bytes(weights)
    artifacts = (
        ModelArtifactPin(
            path="weights.bin",
            bytes=len(weights),
            sha256=hashlib.sha256(weights).hexdigest(),
        ),
    )
    manifest_sha256 = artifact_manifest_sha256(artifacts)
    fields = {
        "name": "fixture_minilm",
        "role": "semantic_embedding",
        "model_id": "fixture/minilm",
        "revision": "a" * 40,
        "artifact_manifest_sha256": manifest_sha256,
    }
    return (
        ModelPin(
            **fields,
            artifacts=artifacts,
            pin_sha256=model_pin_sha256(**fields),
        ),
        snapshot,
    )


def _write_corpus(path: Path) -> None:
    path.write_bytes(FIXTURE_PATH.read_bytes())


def test_tiny_cpu_corpus_to_index_validation_is_deterministic(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path)

    corpus_summary = validate_corpus(corpus_path)
    first = build_tiny_index(corpus_path, tmp_path / "index-a", dimensions=32)
    second = build_tiny_index(corpus_path, tmp_path / "index-b", dimensions=32)

    assert first.corpus == corpus_summary
    assert first == second
    assert first.vectors.record_count == 3
    assert first.vectors.sha256 == (
        "c0355a0d49be41b06819de1a90265ca47fa61df419983227138554e00e0ef8ef"
    )
    assert (tmp_path / "index-a" / "vectors.jsonl").read_bytes() == (
        tmp_path / "index-b" / "vectors.jsonl"
    ).read_bytes()
    assert (tmp_path / "index-a" / "summary.json").read_bytes() == (
        tmp_path / "index-b" / "summary.json"
    ).read_bytes()
    assert validate_index(
        tmp_path / "index-a",
        corpus=corpus_path,
        expected_model_pin=TINY_MODEL_PIN,
    ) == first
    with pytest.raises(IndexValidationError, match="requires the canonical corpus"):
        validate_index(tmp_path / "index-a", require_corpus=True)


def test_index_validation_rejects_output_and_corpus_tampering(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    _write_corpus(corpus_path)
    build_tiny_index(corpus_path, index_dir, dimensions=16)

    vectors_path = index_dir / "vectors.jsonl"
    vectors_path.write_bytes(vectors_path.read_bytes() + b" ")
    with pytest.raises(IndexValidationError, match="byte count mismatch"):
        validate_index(index_dir, corpus=corpus_path)

    build_tiny_index(corpus_path, index_dir, dimensions=16)
    corpus_path.write_bytes(corpus_path.read_bytes().replace(b"Harbor City", b"Harbour City"))
    with pytest.raises(CorpusValidationError, match="paragraph_id does not match content"):
        validate_index(index_dir, corpus=corpus_path)


def test_index_summary_revision_and_hash_are_fail_fast(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    index_dir = tmp_path / "index"
    _write_corpus(corpus_path)
    build_tiny_index(corpus_path, index_dir)
    summary_path = index_dir / "summary.json"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["algorithm_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IndexValidationError, match="algorithm revision"):
        validate_index(index_dir)

    build_tiny_index(corpus_path, index_dir)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["parameters_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IndexValidationError, match="parameter hash mismatch"):
        validate_index(index_dir)


def test_dense_index_is_deterministic_bound_and_queryable(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path)
    model, snapshot = _fixture_model(tmp_path)
    first_dir = tmp_path / "dense-a"
    second_dir = tmp_path / "dense-b"

    first = build_dense_index(
        corpus_path,
        first_dir,
        model=model,
        model_snapshot=snapshot,
        device="cpu",
        batch_size=2,
        encoder=_FixtureEncoder(),
        show_progress_bar=False,
    )
    second = build_dense_index(
        corpus_path,
        second_dir,
        model=model,
        model_snapshot=snapshot,
        device="cpu",
        batch_size=3,
        encoder=_FixtureEncoder(),
        show_progress_bar=False,
    )

    assert first.index_format == "eager.dense-npy-v1"
    assert first.vectors.record_count == 3
    assert first.vectors.sha256 == second.vectors.sha256
    assert first.evidence_ids.sha256 == second.evidence_ids.sha256
    assert validate_dense_index(first_dir, corpus=corpus_path, expected_model_pin=model) == first

    semantic = DenseSemanticIndex.load(
        first_dir,
        corpus=corpus_path,
        model_snapshot=snapshot,
        device="cpu",
        encoder=_FixtureEncoder(),
    )
    requested = tuple(sorted(semantic.evidence_ids))
    scores = semantic.score_many("Where is Northstar Labs?", requested)
    assert set(scores) == set(requested)
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_dense_index_validation_rejects_vector_tampering(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path)
    model, snapshot = _fixture_model(tmp_path)
    index_dir = tmp_path / "dense"
    build_dense_index(
        corpus_path,
        index_dir,
        model=model,
        model_snapshot=snapshot,
        device="cpu",
        encoder=_FixtureEncoder(),
        show_progress_bar=False,
    )
    vectors_path = index_dir / "embeddings.npy"
    vectors_path.write_bytes(vectors_path.read_bytes() + b"tamper")

    with pytest.raises(IndexValidationError, match="vectors byte count mismatch"):
        validate_dense_index(index_dir, corpus=corpus_path)


def test_dense_index_binds_alternate_evidence_text_representation(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(corpus_path)
    corpus = load_corpus_snapshot(corpus_path)
    model, snapshot = _fixture_model(tmp_path)
    evidence_texts = {
        unit.evidence_id: f"resolved: {unit.text}"
        for unit in corpus.evidence_units
    }
    source_sha256 = hashlib.sha256(b"fixture-resolved-texts").hexdigest()
    index_dir = tmp_path / "dense-resolved"

    summary = build_dense_index(
        corpus_path,
        index_dir,
        model=model,
        model_snapshot=snapshot,
        device="cpu",
        encoder=_FixtureEncoder(),
        show_progress_bar=False,
        evidence_texts=evidence_texts,
        evidence_texts_source_sha256=source_sha256,
    )

    assert summary.text_variant == "title_sep_resolved_sentence"
    assert summary.representation is not None
    assert summary.representation.source_sha256 == source_sha256
    assert summary.representation.record_count == len(evidence_texts)
    assert validate_dense_index(index_dir, corpus=corpus_path) == summary

    representation_path = index_dir / "evidence_texts.jsonl"
    representation_path.write_bytes(representation_path.read_bytes() + b" ")
    with pytest.raises(IndexValidationError, match="evidence text byte count mismatch"):
        validate_dense_index(index_dir, corpus=corpus_path)
