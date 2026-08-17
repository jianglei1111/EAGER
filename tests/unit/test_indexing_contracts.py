from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eager.data import make_paragraph_id
from eager.indexing import (
    CorpusRecord,
    CorpusSnapshot,
    CorpusValidationError,
    ModelArtifactPin,
    ModelPin,
    ModelRegistryError,
    artifact_manifest_sha256,
    load_corpus_snapshot,
    load_model_registry,
    model_pin_sha256,
    verify_model_artifact,
    verify_model_snapshot,
)


ROOT = Path(__file__).parents[2]
MODEL_REGISTRY = ROOT / "configs" / "wsdm" / "models" / "registry.json"


def corpus_record(
    *,
    title: str = "Northstar Labs",
    sentences: tuple[str, ...] = ("Northstar Labs is headquartered in Harbor City.",),
) -> CorpusRecord:
    paragraph_id = make_paragraph_id(
        dataset="hotpotqa",
        title=title,
        content=sentences,
    )
    return CorpusRecord(
        paragraph_id=paragraph_id,
        title=title,
        text=" ".join(sentences),
        sentences=sentences,
        source_dataset="HotpotQA",
        source_split="DEV_DISTRACTOR",
    )


def test_corpus_record_enforces_content_id_and_inference_allowlist() -> None:
    record = corpus_record()
    assert record.source_dataset == "hotpotqa"
    assert record.source_split == "dev_distractor"

    payload = record.model_dump(mode="json")
    payload["paragraph_id"] = "hp_par_00000"
    with pytest.raises(ValidationError, match="paragraph_id does not match content"):
        CorpusRecord.model_validate(payload)

    payload = record.model_dump(mode="json")
    payload["gold_supporting_question_ids"] = ["oracle"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorpusRecord.model_validate(payload)


def test_snapshot_maps_records_to_stable_evidence_units() -> None:
    record = corpus_record(
        sentences=(
            "Northstar Labs was founded by Mira Sol.",
            "It is headquartered in Harbor City.",
        )
    )
    snapshot = CorpusSnapshot.from_records([record])

    assert snapshot.summary.paragraph_count == 1
    assert snapshot.summary.evidence_count == 2
    assert snapshot.summary.sentence_count == 2
    assert [(source.dataset, source.split) for source in snapshot.summary.sources] == [
        ("hotpotqa", "dev_distractor")
    ]
    assert [unit.evidence_id for unit in snapshot.evidence_units] == [
        f"{record.paragraph_id}_s0",
        f"{record.paragraph_id}_s1",
    ]
    assert snapshot.evidence_units[0].evidence_id == snapshot.evidence_units[0].sentence_id


def test_paragraph_without_segmentation_has_one_fallback_sentence() -> None:
    text = "A paragraph that has not been segmented."
    record = CorpusRecord(
        paragraph_id=make_paragraph_id(
            dataset="musique",
            title="Synthetic",
            content=text,
        ),
        title="Synthetic",
        text=text,
        sentences=(),
        source_dataset="musique",
        source_split="fixture",
    )
    snapshot = CorpusSnapshot.from_records([record])

    assert snapshot.summary.evidence_count == 1
    assert snapshot.summary.sentence_count == 1
    assert snapshot.evidence_units[0].sentence_id.endswith("_s0")


def test_empty_sentence_slots_preserve_source_positions() -> None:
    sentences = ("The first sentence.", "", "The third sentence.")
    record = CorpusRecord(
        paragraph_id=make_paragraph_id(
            dataset="hotpotqa",
            title="Position Fixture",
            content=sentences,
        ),
        title="Position Fixture",
        text="The first sentence. The third sentence.",
        sentences=sentences,
        source_dataset="hotpotqa",
        source_split="fixture",
    )
    snapshot = CorpusSnapshot.from_records([record])

    assert snapshot.summary.evidence_count == 2
    assert snapshot.summary.sentence_count == 2
    assert snapshot.summary.empty_sentence_count == 1
    assert [unit.sentence_id.rsplit("_s", 1)[1] for unit in snapshot.evidence_units] == ["0", "2"]


def test_whitespace_variants_canonicalize_and_deduplicate_stably() -> None:
    canonical_title = "North Star"
    canonical_sentences = (
        "The first sentence.",
        "",
        "The third sentence.",
    )
    paragraph_id = make_paragraph_id(
        dataset="hotpotqa",
        title=canonical_title,
        content=canonical_sentences,
    )
    canonical = CorpusRecord(
        paragraph_id=paragraph_id,
        title=canonical_title,
        text="The first sentence. The third sentence.",
        sentences=canonical_sentences,
        source_dataset="hotpotqa",
        source_split="fixture",
    )
    variant = CorpusRecord(
        paragraph_id=paragraph_id,
        title="North\u00a0\u00a0Star",
        text="The  first sentence.\u00a0 The third  sentence.",
        sentences=(
            "The  first sentence.",
            "\u00a0",
            "The third\u00a0 sentence.",
        ),
        source_dataset="HOTPOTQA",
        source_split="FIXTURE",
    )

    assert variant == canonical
    snapshot = CorpusSnapshot.from_records([variant, canonical])
    assert snapshot.records == (canonical,)
    assert snapshot.summary.paragraph_count == 1
    assert snapshot.summary.empty_sentence_count == 1


def test_corpus_loader_rejects_duplicate_ids_and_blank_lines(tmp_path: Path) -> None:
    record = corpus_record()
    line = json.dumps(record.model_dump(mode="json"), separators=(",", ":"))
    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(f"{line}\n{line}\n", encoding="utf-8", newline="\n")
    with pytest.raises(CorpusValidationError, match="duplicate paragraph_id"):
        load_corpus_snapshot(duplicate_path)

    blank_path = tmp_path / "blank.jsonl"
    blank_path.write_text(f"{line}\n\n", encoding="utf-8", newline="\n")
    with pytest.raises(CorpusValidationError, match="cannot contain blank lines"):
        load_corpus_snapshot(blank_path)


def test_model_registry_pins_revision_and_entry_hash(tmp_path: Path) -> None:
    registry = load_model_registry(MODEL_REGISTRY)
    model = registry.resolve("minilm_l6_v2", revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
    assert model.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    lock = json.loads((ROOT / "requirements" / "models.lock.json").read_text(encoding="utf-8"))
    assert {
        (item.model_id, item.role, item.revision) for item in registry.models
    } == {
        (item["model_id"], item["role"], item["revision"])
        for item in lock["huggingface_models"]
    }

    with pytest.raises(ModelRegistryError, match="revision mismatch"):
        registry.resolve("minilm_l6_v2", revision="0" * 40)
    with pytest.raises(ModelRegistryError, match="pin hash mismatch"):
        registry.resolve("minilm_l6_v2", pin_sha256="0" * 64)

    payload = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
    payload["models"][0]["revision"] = "main"
    bad_revision = tmp_path / "bad-revision.json"
    bad_revision.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelRegistryError, match="invalid model registry"):
        load_model_registry(bad_revision)

    payload = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
    payload["models"][0]["pin_sha256"] = "0" * 64
    bad_hash = tmp_path / "bad-hash.json"
    bad_hash.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelRegistryError, match="pin_sha256 mismatch"):
        load_model_registry(bad_hash)

    payload = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
    payload["models"][0]["artifacts"][0]["sha256"] = "0" * 64
    bad_manifest = tmp_path / "bad-manifest.json"
    bad_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelRegistryError, match="artifact_manifest_sha256 mismatch"):
        load_model_registry(bad_manifest)


def test_model_artifact_hash_is_fail_fast(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"synthetic model fixture")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert verify_model_artifact(artifact, expected_sha256=expected) == expected
    with pytest.raises(ModelRegistryError, match="artifact hash mismatch"):
        verify_model_artifact(artifact, expected_sha256="0" * 64)

    with pytest.raises(ValidationError, match="relative POSIX path"):
        ModelArtifactPin(path="C:escape", bytes=1, sha256="0" * 64)

    upper = ModelArtifactPin(path="Config.json", bytes=1, sha256="0" * 64)
    lower = ModelArtifactPin(path="config.json", bytes=1, sha256="0" * 64)
    manifest_sha256 = artifact_manifest_sha256([upper, lower])
    fields = {
        "name": "case_conflict",
        "role": "test_embedding",
        "model_id": "eager/case-conflict",
        "revision": "1" * 40,
        "artifact_manifest_sha256": manifest_sha256,
    }
    with pytest.raises(ValidationError, match="conflict case-insensitively"):
        ModelPin(
            **fields,
            artifacts=(upper, lower),
            pin_sha256=model_pin_sha256(**fields),
        )


def test_model_snapshot_binds_revision_size_and_file_hash(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    artifact_path = snapshot_dir / "weights.bin"
    artifact_path.write_bytes(b"synthetic model fixture")
    artifact = ModelArtifactPin(
        path="weights.bin",
        bytes=artifact_path.stat().st_size,
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    )
    manifest_sha256 = artifact_manifest_sha256([artifact])
    fields = {
        "name": "synthetic_encoder",
        "role": "test_embedding",
        "model_id": "eager/synthetic-encoder",
        "revision": "1" * 40,
        "artifact_manifest_sha256": manifest_sha256,
    }
    model = ModelPin(
        **fields,
        artifacts=(artifact,),
        pin_sha256=model_pin_sha256(**fields),
    )

    assert verify_model_snapshot(
        snapshot_dir,
        model=model,
        revision=model.revision,
    ) == manifest_sha256
    with pytest.raises(ModelRegistryError, match="revision mismatch"):
        verify_model_snapshot(snapshot_dir, model=model, revision="2" * 40)

    artifact_path.write_bytes(b"synthetic model fixturE")
    with pytest.raises(ModelRegistryError, match="artifact hash mismatch"):
        verify_model_snapshot(snapshot_dir, model=model, revision=model.revision)
