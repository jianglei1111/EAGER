from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from eager.runtime.checkpoint import CheckpointMismatchError, CheckpointStore, ProvenanceBinding
from eager.runtime.io import atomic_write_json, read_json
from eager.runtime.logging import configure_logging, redact
from eager.runtime.manifest import git_snapshot
from eager.runtime.paths import PathContractError, resolve_path
from eager.runtime.regression import ComparisonPolicy, assert_matches, compare, stable_rank


def test_atomic_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert read_json(path) == {"a": 1, "b": 2}
    assert not list(path.parent.glob("*.tmp"))


def test_path_contract_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(PathContractError):
        resolve_path("../escape.json", base=tmp_path)


def test_logging_redacts_nested_secrets() -> None:
    fake_bearer = "Bearer " + "abcdefghijklmnop"
    assert redact(
        {
            "api_key": "secret-value",
            "api_key_env": "OPENAI_API_KEY",
            "input_tokens": 12,
            "output_tokens": 3,
            "safe": [fake_bearer],
        }
    ) == {
        "api_key": "<redacted>",
        "api_key_env": "OPENAI_API_KEY",
        "input_tokens": 12,
        "output_tokens": 3,
        "safe": ["<redacted>"],
    }
    stream = io.StringIO()
    logger = configure_logging(stream=stream, logger_name="eager-test")
    fake_token = "sk-" + "test-secret-value-abcdefghijkl"
    logger.info("token %s", fake_token)
    output = stream.getvalue()
    assert "abcdefghijkl" not in output
    assert "<redacted>" in output


def test_checkpoint_is_idempotent_and_bound_to_provenance(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints.jsonl")
    binding = ProvenanceBinding("a", "b", "c", "d", "e")
    first = store.append(checkpoint_id="q1:0", stage="retrieval", binding=binding, state={"rank": 1})
    second = store.append(checkpoint_id="q1:0", stage="retrieval", binding=binding, state={"rank": 1})
    assert first == second
    assert store.resume("q1:0", binding) == first
    with pytest.raises(CheckpointMismatchError):
        store.resume("q1:0", ProvenanceBinding("x", "b", "c", "d", "e"))


def test_regression_comparator_uses_tolerance_and_stable_order() -> None:
    assert not compare({"score": 1.0}, {"score": 1.00001}, ComparisonPolicy(1e-3, 0.0))
    assert compare({"score": 1.0}, {"score": 1.1})
    with pytest.raises(AssertionError):
        assert_matches([1, 2], [1, 3])
    rows = [{"id": "b", "score": 1.0}, {"id": "a", "score": 1.0}]
    assert [row["id"] for row in stable_rank(rows, score_field="score", id_field="id")] == ["a", "b"]


def test_git_snapshot_reports_unavailable_outside_a_clone(tmp_path: Path) -> None:
    assert git_snapshot(tmp_path) == {
        "status": "unavailable",
        "commit": None,
        "branch": None,
        "dirty": None,
    }
