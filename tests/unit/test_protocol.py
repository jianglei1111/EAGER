from __future__ import annotations

import json
from pathlib import Path

import pytest

from eager.protocol import ProtocolError, load_protocol, migrate_protocol, protocol_sha256, validate_protocol


def legacy_protocol() -> dict[str, object]:
    return {
        "schema_version": 1,
        "objective": "test",
        "datasets": {
            "hotpotqa": {
                "question_config": "configs/questions.json",
                "processed_sentences": "data/sentences.jsonl",
                "index_dir": "data/indexes/hotpotqa",
                "primary_support_unit": "paragraph",
                "secondary_support_unit": "sentence",
            }
        },
        "evaluation": {
            "frozen_test_queries": 1000,
            "development_queries": "200-300 disjoint questions",
            "timing_queries": 20,
            "candidate_budgets": [50, 100],
            "final_evidence_budgets": [5, 10],
            "corpus_scales": [1, 2],
            "random_seed": 7,
        },
        "candidate_variants": ["full"],
        "refinement_variants": ["hierarchical"],
        "policy_variants": ["rule"],
        "required_metrics": ["answer_f1"],
    }


def test_v1_migration_is_typed_and_explicit() -> None:
    migrated = migrate_protocol(legacy_protocol())
    assert migrated["schema_version"] == 2
    assert migrated["evaluation"]["development"]["default_queries"] == 200
    protocol = validate_protocol(migrated)
    assert protocol.execution_profiles["wsdm_strict"].oracle_annotations_allowed is False
    assert "raw_question_id" in protocol.execution_profiles["wsdm_strict"].forbidden_controller_fields
    assert len(protocol_sha256(protocol)) == 64


def test_protocol_paths_reject_machine_absolute_paths() -> None:
    migrated = migrate_protocol(legacy_protocol())
    migrated["datasets"]["hotpotqa"]["index_dir"] = "D:\\private\\index"
    with pytest.raises(ProtocolError):
        validate_protocol(migrated)


def test_load_current_protocol() -> None:
    path = Path("wsdm/config/experiment_protocol.json")
    protocol = load_protocol(path)
    assert protocol.schema_version == 2
    assert set(protocol.datasets) == {"hotpotqa", "2wiki", "musique"}
    encoded = json.dumps(protocol.model_dump(mode="json"), sort_keys=True)
    assert "development_split" in encoded
