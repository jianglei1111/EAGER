from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from eager.data import (
    DatasetAdapterError,
    HotpotQAAdapter,
    MusiqueAdapter,
    TwoWikiAdapter,
    make_paragraph_id,
    make_sentence_id,
    resolve_registered_split,
    validate_split_overlaps,
)


ROOT = Path(__file__).parents[2]
FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "data" / "adapter_rows.json"
ROOT = Path(__file__).parents[2]
SPLIT_REGISTRY = ROOT / "wsdm" / "config" / "split_registry.json"


@pytest.fixture(scope="module")
def adapter_rows() -> dict[str, dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _id_hash(question_ids: list[str], *, ordered: bool) -> str:
    values = question_ids if ordered else sorted(question_ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _write_registry(tmp_path: Path, *, source_path: str) -> Path:
    ids = ["hp_fixture_1"]
    ids_path = tmp_path / "splits" / "fixture.ids.txt"
    ids_path.parent.mkdir(parents=True)
    ids_path.write_text("hp_fixture_1\n", encoding="utf-8", newline="\n")
    registry = {
        "schema_version": 2,
        "raw_sources": {
            "hotpotqa": {
                "path": source_path,
                "format": "json_array",
                "id_field": "_id",
                "canonical_id": "identity",
                "hash_mode": "byte_exact",
                "expected_bytes": 1,
                "sha256": "0" * 64,
            }
        },
        "frozen_profiles": {},
        "development_splits": {
            "fixture": {
                "dataset": "hotpotqa",
                "source_key": "hotpotqa",
                "ids_path": "splits/fixture.ids.txt",
                "expected_count": 1,
                "ordered_id_sha256": _id_hash(ids, ordered=True),
                "set_id_sha256": _id_hash(ids, ordered=False),
            }
        },
    }
    registry_path = tmp_path / "wsdm" / "config" / "split_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return registry_path


def test_content_ids_are_normalized_stable_and_content_sensitive() -> None:
    normalized = make_paragraph_id(
        dataset="musique",
        title="Atlas",
        content="Nia Reed  was born in Bellhaven.",
    )
    equivalent = make_paragraph_id(
        dataset=" MUSIQUE ",
        title="Ａtlas ",
        content=" Nia Reed was born in Bellhaven. ",
    )
    changed = make_paragraph_id(
        dataset="musique",
        title="Atlas",
        content="Nia Reed was born in another city.",
    )

    assert normalized == equivalent
    assert normalized != changed
    assert make_sentence_id(normalized, 3) == f"{normalized}_s3"
    assert len(normalized.removeprefix("mu_par_")) == 32


def test_handles_and_support_ids_do_not_depend_on_context_order(
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    original = deepcopy(adapter_rows["hotpotqa"])
    reordered = deepcopy(original)
    reordered["context"] = list(reversed(reordered["context"]))
    adapter = HotpotQAAdapter(split="fixture")

    original_handle, _, original_annotations = adapter.adapt_row(original)
    reordered_handle, _, reordered_annotations = adapter.adapt_row(reordered)

    assert original_handle == reordered_handle
    assert (
        original_annotations.supporting_sentence_ids
        == reordered_annotations.supporting_sentence_ids
    )
    assert (
        original_annotations.supporting_paragraph_ids
        == reordered_annotations.supporting_paragraph_ids
    )


def test_duplicate_source_ids_are_rejected_even_outside_selection(
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    first = deepcopy(adapter_rows["hotpotqa"])
    duplicate = deepcopy(first)
    adapter = HotpotQAAdapter(split="fixture")

    with pytest.raises(DatasetAdapterError, match="duplicate source question ID"):
        adapter.adapt_rows([first, duplicate], question_ids=["hp_fixture_1"])


@pytest.mark.parametrize(
    ("dataset", "adapter_type"),
    [
        ("hotpotqa", HotpotQAAdapter),
        ("2wiki", TwoWikiAdapter),
        ("musique", MusiqueAdapter),
    ],
)
def test_canonical_corpus_is_invariant_to_evaluation_annotations(
    adapter_rows: dict[str, dict[str, object]],
    dataset: str,
    adapter_type: type,
) -> None:
    row = deepcopy(adapter_rows[dataset])
    mutated = deepcopy(row)
    if dataset == "hotpotqa":
        mutated.update(
            answer="a different gold answer",
            supporting_facts=[["absent gold title", 999]],
            type="comparison",
        )
    elif dataset == "2wiki":
        mutated.update(
            answer="a different gold answer",
            supporting_facts="not annotation JSON",
            evidences="not relation JSON",
            type="inference",
        )
    else:
        mutated.update(
            answer="a different gold answer",
            answer_aliases=["another alias"],
            answerable=False,
            question_decomposition=[],
        )
        for paragraph in mutated["paragraphs"]:
            paragraph["is_supporting"] = not paragraph["is_supporting"]

    adapter = adapter_type(split="fixture")
    assert adapter.build_corpus_snapshot([row]) == adapter.build_corpus_snapshot([mutated])


def test_musique_jsonl_reader_wraps_file_errors(tmp_path: Path) -> None:
    malformed = tmp_path / "musique.jsonl"
    malformed.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(DatasetAdapterError, match="invalid JSONL"):
        MusiqueAdapter(split="fixture").load(malformed)


@pytest.mark.parametrize(
    "path_kind",
    ["parent_escape", "absolute"],
)
def test_registry_rejects_source_paths_outside_repo(tmp_path: Path, path_kind: str) -> None:
    source_path = (
        "../outside.json"
        if path_kind == "parent_escape"
        else str((tmp_path.parent / "outside" / "hotpotqa.json").resolve())
    )
    registry_path = _write_registry(tmp_path, source_path=source_path)

    with pytest.raises(DatasetAdapterError, match="repository-relative|escapes the repository"):
        resolve_registered_split(registry_path, "fixture", repo_root=tmp_path)


@pytest.mark.parametrize(
    ("split_name", "expected_revision", "expected_license"),
    [
        ("hotpotqa_dev200", "hotpot_dev_distractor_v1", "CC-BY-SA-4.0"),
        ("2wiki_dev200", "612bc5039a457880d9e7d84c3b0a4cf154b70e4f", "Apache-2.0"),
        ("musique_dev200", "22873a405dd809893b22ada0b499299fb612d2df", "CC-BY-4.0"),
    ],
)
def test_v3_registry_exposes_non_floating_source_and_license_provenance(
    split_name: str,
    expected_revision: str,
    expected_license: str,
) -> None:
    split = resolve_registered_split(SPLIT_REGISTRY, split_name, repo_root=ROOT)

    assert split.registry_schema_version == 3
    assert split.source_revision == expected_revision
    assert split.license_spdx == expected_license
    assert split.source_url is not None
    assert "/main/" not in split.source_url
    assert "/master/" not in split.source_url
    assert split.source_artifact_verification
    assert split.source_repository_revision is not None
    assert len(split.source_repository_revision) == 40
    assert split.license_reference is not None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xef\xbb\xbfhp_fixture_1\n", "must not contain a UTF-8 BOM"),
        (b"hp_fixture_1\r\n", "must use LF line endings"),
        (b"hp_fixture_1", "must end with a final LF"),
    ],
)
def test_registry_enforces_declared_id_file_serialization(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    registry_path = _write_registry(tmp_path, source_path="data/raw/hotpotqa.json")
    (tmp_path / "splits" / "fixture.ids.txt").write_bytes(payload)

    with pytest.raises(DatasetAdapterError, match=message):
        resolve_registered_split(registry_path, "fixture", repo_root=tmp_path)


def test_tracked_split_overlap_contract_is_executable(tmp_path: Path) -> None:
    registry_path = ROOT / "wsdm" / "config" / "split_registry.json"
    results = validate_split_overlaps(registry_path, repo_root=ROOT)
    assert len(results) == 3
    assert all(result["actual_intersection"] == 0 for result in results)

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["overlap_constraints"][0]["expected_intersection"] = 1
    tampered = tmp_path / "split_registry.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetAdapterError, match="split overlap mismatch"):
        validate_split_overlaps(tampered, repo_root=ROOT)
