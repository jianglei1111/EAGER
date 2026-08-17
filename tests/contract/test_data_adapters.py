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
    encode_relation,
    load_registered_split,
    resolve_registered_split,
)
from eager.schemas import FORBIDDEN_INFERENCE_FIELDS, assert_annotation_isolation


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "data" / "adapter_rows.json"


@pytest.fixture(scope="module")
def adapter_rows() -> dict[str, dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("dataset", "adapter_type", "raw_id"),
    [
        ("hotpotqa", HotpotQAAdapter, "hp_fixture_1"),
        ("2wiki", TwoWikiAdapter, "tw_fixture_1"),
        ("musique", MusiqueAdapter, "2hop__101_202"),
    ],
)
def test_adapters_physically_isolate_engine_payload(
    adapter_rows: dict[str, dict[str, object]],
    dataset: str,
    adapter_type: type,
    raw_id: str,
) -> None:
    handle, payload, annotation = adapter_type(split="fixture").adapt_row(adapter_rows[dataset])

    serialized = payload.model_dump(mode="json")
    assert serialized == {"dataset": dataset, "question": adapter_rows[dataset]["question"]}
    assert raw_id not in json.dumps(serialized)
    assert not (set(serialized) & FORBIDDEN_INFERENCE_FIELDS)
    assert_annotation_isolation(serialized)
    assert handle.raw_dataset_id == raw_id
    assert annotation.handle_id == handle.handle_id


def test_hotpotqa_maps_sentence_support(adapter_rows: dict[str, dict[str, object]]) -> None:
    _, _, annotation = HotpotQAAdapter(split="fixture").adapt_row(adapter_rows["hotpotqa"])

    assert annotation.answer == "Harbor City"
    assert annotation.gold_type == "bridge"
    assert annotation.hop_count == 2
    assert annotation.supporting_titles == ("Mira Sol", "Northstar Labs")
    assert len(annotation.supporting_paragraph_ids) == 2
    assert len(annotation.supporting_sentence_ids) == 2
    assert annotation.supporting_sentence_ids[0].endswith("_s0")


def test_adapter_rejects_out_of_range_support_without_rewriting_gold(
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    row = deepcopy(adapter_rows["hotpotqa"])
    row["supporting_facts"][0][1] = 902

    with pytest.raises(DatasetAdapterError, match="sentence index 902 is out of range"):
        HotpotQAAdapter(split="fixture").adapt_row(row)


def test_adapter_rejects_ambiguous_duplicate_support_title(
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    row = deepcopy(adapter_rows["hotpotqa"])
    row["context"].append(["Mira Sol", ["A different paragraph with the same title."]])

    with pytest.raises(DatasetAdapterError, match="title is ambiguous across 2 paragraphs"):
        HotpotQAAdapter(split="fixture").adapt_row(row)


def test_json_reader_errors_use_public_adapter_error(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_text("{not-json", encoding="utf-8")

    with pytest.raises(DatasetAdapterError, match="cannot read hotpotqa source"):
        HotpotQAAdapter(split="fixture").load(source)


def test_2wiki_decodes_json_columns_and_relations(adapter_rows: dict[str, dict[str, object]]) -> None:
    _, _, annotation = TwoWikiAdapter(split="fixture").adapt_row(adapter_rows["2wiki"])

    assert annotation.gold_type == "compositional"
    assert annotation.hop_count == 2
    assert annotation.gold_relations == (
        encode_relation(("The Silent Map", "director", "Elian Voss")),
        encode_relation(("Elian Voss", "mother", "Rhea Voss")),
    )
    assert len(annotation.supporting_sentence_ids) == 2


def test_musique_preserves_duplicate_title_paragraph_identity(
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    adapter = MusiqueAdapter(split="fixture")
    handle, _, annotation = adapter.adapt_row(adapter_rows["musique"])

    assert adapter.canonical_question_id(adapter_rows["musique"]) == "musique_2hop__101_202"
    assert handle.raw_dataset_id == "2hop__101_202"
    assert len(annotation.supporting_paragraph_ids) == 2
    assert len(set(annotation.supporting_paragraph_ids)) == 2
    assert annotation.supporting_titles == ("Atlas",)
    assert annotation.answer_aliases == ("Bellhaven City",)
    assert annotation.gold_type == "2hop"
    assert annotation.hop_count == 2
    first, second = annotation.official_decomposition
    assert first.step_id == "101"
    assert second.depends_on == ("101",)
    assert second.supporting_paragraph_ids == (annotation.supporting_paragraph_ids[1],)
    assert second.relation == "place of birth"


def test_musique_builds_sentence_level_corpus_with_paragraph_level_support(
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    row = dict(adapter_rows["musique"])
    paragraphs = [dict(item) for item in row["paragraphs"]]
    paragraphs[0]["paragraph_text"] = "Atlas was written by Nia Reed. It won an award."
    row["paragraphs"] = paragraphs
    adapter = MusiqueAdapter(split="ans_dev")

    batch = adapter.adapt_rows([row])
    corpus = adapter.build_corpus_snapshot([row])
    first = next(record for record in corpus.records if record.title == "Atlas" and len(record.sentences) == 2)
    annotation = next(iter(batch.evaluation.by_handle_id.values()))

    assert first.sentences == ("Atlas was written by Nia Reed.", "It won an award.")
    assert first.paragraph_id in annotation.supporting_paragraph_ids
    assert {unit.paragraph_id for unit in corpus.evidence_units} >= set(
        annotation.supporting_paragraph_ids
    )


@pytest.mark.parametrize(
    ("dataset", "adapter_type"),
    [
        ("hotpotqa", HotpotQAAdapter),
        ("2wiki", TwoWikiAdapter),
        ("musique", MusiqueAdapter),
    ],
)
def test_adapters_build_annotation_free_canonical_corpus(
    adapter_rows: dict[str, dict[str, object]],
    dataset: str,
    adapter_type: type,
) -> None:
    adapter = adapter_type(split="fixture")
    row = adapter_rows[dataset]
    snapshot = adapter.build_corpus_snapshot([row])
    _, _, annotations = adapter.adapt_row(row)

    assert snapshot.summary.sources[0].dataset == dataset
    assert snapshot.summary.sources[0].split == "fixture"
    assert snapshot.evidence_units
    assert all(unit.evidence_id == unit.sentence_id for unit in snapshot.evidence_units)
    assert set(annotations.supporting_paragraph_ids) <= {
        record.paragraph_id for record in snapshot.records
    }
    if annotations.supporting_sentence_ids:
        assert set(annotations.supporting_sentence_ids) <= {
            unit.sentence_id for unit in snapshot.evidence_units
        }
    for record in snapshot.records:
        assert not (
            set(record.model_dump(mode="json"))
            & {"answer", "gold_type", "supporting_facts", "question_id"}
        )


def test_corpus_builder_stably_deduplicates_shared_paragraphs(
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    first = deepcopy(adapter_rows["hotpotqa"])
    second = deepcopy(first)
    second["_id"] = "hp_fixture_2"
    second["question"] = "Another question sharing the same public context?"

    snapshot = HotpotQAAdapter(split="fixture").build_corpus_snapshot([first, second])
    assert len(snapshot.records) == len(first["context"])


def test_batch_selection_uses_registry_order_and_rejects_missing_ids(
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    first = dict(adapter_rows["hotpotqa"])
    second = dict(adapter_rows["hotpotqa"])
    second["_id"] = "hp_fixture_2"
    second["question"] = "A second fixture question?"
    adapter = HotpotQAAdapter(split="fixture")

    batch = adapter.adapt_rows(
        [first, second],
        question_ids=["hp_fixture_2", "hp_fixture_1"],
    )
    assert [handle.raw_dataset_id for handle in batch.inference.handles] == [
        "hp_fixture_2",
        "hp_fixture_1",
    ]
    assert [payload.question for payload in batch.inference.engine_payloads()] == [
        "A second fixture question?",
        first["question"],
    ]
    assert set(batch.evaluation.by_handle_id) == {
        handle.handle_id for handle in batch.inference.handles
    }

    with pytest.raises(DatasetAdapterError, match="missing 1 requested"):
        adapter.adapt_rows([first], question_ids=["absent"])
    with pytest.raises(DatasetAdapterError, match="contains duplicates"):
        adapter.adapt_rows([first], question_ids=["hp_fixture_1", "hp_fixture_1"])


def test_2wiki_parquet_reader_contract(
    tmp_path: Path,
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "dev.parquet"
    pd.DataFrame([adapter_rows["2wiki"]]).to_parquet(path, index=False)

    batch = TwoWikiAdapter(split="fixture").load(path, question_ids=["tw_fixture_1"])
    assert batch.inference.payloads[0].question == adapter_rows["2wiki"]["question"]


def _id_hash(ids: list[str], *, ordered: bool) -> str:
    values = ids if ordered else sorted(ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def test_registered_split_validates_id_and_source_hashes(
    tmp_path: Path,
    adapter_rows: dict[str, dict[str, object]],
) -> None:
    source_path = tmp_path / "data" / "raw" / "hotpotqa.json"
    ids_path = tmp_path / "splits" / "dev.ids.txt"
    registry_path = tmp_path / "wsdm" / "config" / "split_registry.json"
    source_path.parent.mkdir(parents=True)
    ids_path.parent.mkdir(parents=True)
    registry_path.parent.mkdir(parents=True)
    source_bytes = (json.dumps([adapter_rows["hotpotqa"]], separators=(",", ":")) + "\n").encode()
    source_path.write_bytes(source_bytes)
    ids_path.write_text("hp_fixture_1\n", encoding="utf-8", newline="\n")
    ids = ["hp_fixture_1"]
    registry = {
        "schema_version": 2,
        "raw_sources": {
            "hotpotqa": {
                "path": "data/raw/hotpotqa.json",
                "format": "json_array",
                "id_field": "_id",
                "canonical_id": "identity",
                "hash_mode": "byte_exact",
                "expected_bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        },
        "frozen_profiles": {},
        "development_splits": {
            "hotpot_fixture": {
                "dataset": "hotpotqa",
                "source_key": "hotpotqa",
                "ids_path": "splits/dev.ids.txt",
                "expected_count": 1,
                "ordered_id_sha256": _id_hash(ids, ordered=True),
                "set_id_sha256": _id_hash(ids, ordered=False),
            }
        },
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    registered = resolve_registered_split(registry_path, "hotpot_fixture", repo_root=tmp_path)
    assert registered.question_ids == ("hp_fixture_1",)
    batch = load_registered_split(registry_path, "hotpot_fixture", repo_root=tmp_path)
    assert batch.inference.handles[0].split == "hotpot_fixture"

    tampered_source = bytearray(source_bytes)
    tampered_source[0] = ord("{")
    source_path.write_bytes(tampered_source)
    with pytest.raises(DatasetAdapterError, match="source hash mismatch"):
        load_registered_split(registry_path, "hotpot_fixture", repo_root=tmp_path)
    source_path.write_bytes(source_bytes)

    ids_path.write_text("tampered\n", encoding="utf-8", newline="\n")
    with pytest.raises(DatasetAdapterError, match="ordered ID hash mismatch"):
        resolve_registered_split(registry_path, "hotpot_fixture", repo_root=tmp_path)
