from __future__ import annotations

import json
from pathlib import Path

import pytest

from eager.config import (
    ConfigResolutionError,
    SecretInTrackedConfigError,
    assert_no_secret_fields,
    resolve_config,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def base_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "runtime": {"profile": "development", "random_seed": 1, "log_level": "INFO"},
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.invalid/v1",
            "model": "base-model",
            "api_key_env": "OPENAI_API_KEY",
            "temperature": 0.0,
            "max_completion_tokens": 100,
            "timeout_seconds": 30,
            "max_retries": 1,
            "retry_backoff_seconds": 0,
        },
        "paths": {
            "artifact_root": {"path": "artifacts/test", "base": "repo"},
            "experiment_protocol": {"path": "protocol.json", "base": "config"},
            "split_registry": {"path": "wsdm/config/split_registry.json", "base": "repo"},
        },
    }


def test_config_precedence_and_secret_separation(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    local = tmp_path / "local.json"
    write_json(base, base_payload())
    write_json(local, {"model": "local-model", "api_key": "private-key", "timeout": 20})

    resolved = resolve_config(
        base_path=base,
        local_path=local,
        repo_root=tmp_path,
        environment={"EAGER__LLM__MODEL": "env-model", "EAGER__RUNTIME__RANDOM_SEED": "7"},
        cli_entries=("llm.model=cli-model", "llm.max_retries=4"),
    )

    assert resolved.config.llm.model == "cli-model"
    assert resolved.config.llm.timeout_seconds == 20
    assert resolved.config.runtime.random_seed == 7
    assert resolved.config.llm.max_retries == 4
    assert resolved.secret("llm.api_key") == "private-key"
    assert "private-key" not in json.dumps(resolved.public_dict())
    metadata = resolved.public_metadata()
    assert metadata["environment_override_names"] == [
        "EAGER__LLM__MODEL",
        "EAGER__RUNTIME__RANDOM_SEED",
    ]
    assert metadata["cli_override_paths"] == ["llm.model", "llm.max_retries"]


def test_tracked_base_rejects_secret(tmp_path: Path) -> None:
    payload = base_payload()
    assert isinstance(payload["llm"], dict)
    payload["llm"]["api_key"] = "must-not-be-tracked"
    base = tmp_path / "base.json"
    write_json(base, payload)
    with pytest.raises(SecretInTrackedConfigError):
        resolve_config(base_path=base, repo_root=tmp_path, environment={})


def test_paths_are_bound_to_declared_base(tmp_path: Path) -> None:
    base = tmp_path / "config" / "base.json"
    base.parent.mkdir()
    write_json(base, base_payload())
    resolved = resolve_config(base_path=base, repo_root=tmp_path, environment={})
    assert resolved.resolve_path(resolved.config.paths.artifact_root) == tmp_path / "artifacts" / "test"
    assert resolved.resolve_path(resolved.config.paths.experiment_protocol) == base.parent / "protocol.json"
    assert resolved.resolve_path(resolved.config.paths.dataset_anomalies) == (
        tmp_path / "wsdm" / "config" / "dataset_anomalies.json"
    )
    assert resolved.resolve_path(resolved.config.paths.model_registry) == (
        tmp_path / "configs" / "wsdm" / "models" / "registry.json"
    )


def test_unknown_fields_fail_closed(tmp_path: Path) -> None:
    payload = base_payload()
    payload["unexpected"] = True
    base = tmp_path / "base.json"
    write_json(base, payload)
    with pytest.raises(ConfigResolutionError):
        resolve_config(base_path=base, repo_root=tmp_path, environment={})


def test_retrieval_config_builds_comparable_kernel_and_budget(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    write_json(base, base_payload())
    resolved = resolve_config(
        base_path=base,
        repo_root=tmp_path,
        environment={},
        cli_entries=(
            "retrieval.entity_expansion_enabled=false",
            "retrieval.hierarchical_refinement_enabled=false",
            "retrieval.rq_evidence_window_mode=latest_balanced",
            "retrieval.retained_anchor_scope=latest_step",
            "retrieval.retained_text_anchor_limit=0",
            "retrieval.pre_score_candidate_limit=100",
        ),
    )

    retrieval = resolved.config.retrieval
    assert retrieval.kernel_config().entity_expansion_enabled is False
    assert retrieval.kernel_config().hierarchical_refinement_enabled is False
    assert retrieval.rq_evidence_window_mode == "latest_balanced"
    assert retrieval.retained_anchor_scope == "latest_step"
    assert retrieval.kernel_config().retained_anchor_scope == "latest_step"
    assert retrieval.retained_text_anchor_limit == 0
    assert retrieval.kernel_config().retained_text_anchor_limit == 0
    assert retrieval.pre_score_candidate_limit == 100
    assert retrieval.kernel_config().pre_score_candidate_limit == 100
    assert retrieval.effective_for_signal_mode().evidence_derived_state_anchors_enabled is True
    assert (
        retrieval.effective_for_signal_mode()
        .kernel_config()
        .evidence_derived_state_anchors_enabled
        is True
    )
    budget = retrieval.budget_contract()
    assert budget.max_iterations == retrieval.max_iterations
    assert budget.max_candidates_per_question == retrieval.candidate_budget


def test_retrieval_dataset_profile_resolves_without_inference_metadata(
    tmp_path: Path,
) -> None:
    payload = base_payload()
    payload["retrieval"] = {
        "evidence_derived_state_anchors_enabled": True,
        "max_iterations": 2,
        "candidate_limit": 50,
        "candidate_budget": 100,
        "paragraph_limit": 5,
        "paragraph_budget": 10,
        "rerank_limit": 20,
        "rerank_budget": 40,
        "evidence_limit": 10,
        "dataset_profiles": {
            "musique": {
                "evidence_derived_state_anchors_enabled": False,
                "max_iterations": 4,
                "candidate_budget": 200,
                "paragraph_budget": 20,
                "rerank_budget": 80,
                "evidence_limit": 20,
            }
        },
    }
    base = tmp_path / "base.json"
    write_json(base, payload)
    resolved = resolve_config(base_path=base, repo_root=tmp_path, environment={})

    base_retrieval = resolved.config.retrieval
    effective = base_retrieval.effective_for_dataset("musique")

    assert base_retrieval.max_iterations == 2
    assert base_retrieval.evidence_limit == 10
    assert base_retrieval.evidence_derived_state_anchors_enabled is True
    assert effective.max_iterations == 4
    assert effective.evidence_limit == 20
    assert effective.evidence_derived_state_anchors_enabled is False
    assert effective.kernel_config().evidence_derived_state_anchors_enabled is False
    assert effective.budget_contract().max_candidates_per_question == 200
    assert effective.effective_for_dataset("unknown") == effective


def test_subquestion_chain_budget_reserves_one_audited_retry_step(
    tmp_path: Path,
) -> None:
    payload = base_payload()
    payload["retrieval"] = {
        "signal_mode": "subquestion_chain",
        "max_iterations": 4,
        "max_chain_steps": 6,
        "candidate_limit": 50,
        "candidate_budget": 200,
        "paragraph_limit": 5,
        "paragraph_budget": 20,
        "rerank_limit": 20,
        "final_rerank_limit": 40,
        "rerank_budget": 120,
        "evidence_limit": 30,
        "chain_break_retry_limit": 1,
    }
    base = tmp_path / "base.json"
    write_json(base, payload)
    resolved = resolve_config(base_path=base, repo_root=tmp_path, environment={})

    budget = resolved.config.retrieval.budget_contract()
    effective = resolved.config.retrieval.effective_for_signal_mode()

    assert budget.max_iterations == 7
    assert budget.max_candidates_per_question == 350
    assert budget.max_parent_units_per_question == 35
    assert budget.max_rerank_pairs_per_question == 180
    assert budget.max_llm_calls_per_question == 11
    assert effective.evidence_derived_state_anchors_enabled is False
    assert effective.kernel_config().evidence_derived_state_anchors_enabled is False
    assert resolved.config.retrieval.evidence_derived_state_anchors_enabled is True


def test_dataset_profile_applies_state_anchor_policy_to_both_signals(
    tmp_path: Path,
) -> None:
    payload = base_payload()
    payload["retrieval"] = {
        "signal_mode": "retrieval_question",
        "hierarchical_refinement_enabled": True,
        "rerank_limit": 20,
        "final_rerank_limit": 40,
        "rerank_budget": 120,
        "evidence_limit": 30,
        "dataset_profiles": {
            "musique": {
                "evidence_derived_state_anchors_enabled": False,
            }
        },
    }
    base = tmp_path / "base.json"
    write_json(base, payload)
    resolved = resolve_config(base_path=base, repo_root=tmp_path, environment={})

    rq = resolved.config.retrieval.effective_for_dataset("musique")
    assert rq.evidence_derived_state_anchors_enabled is False
    assert rq.hierarchical_refinement_enabled is True
    assert rq.rerank_limit == 20
    assert rq.final_rerank_limit == 40
    assert rq.rerank_budget == 120

    sc = resolved.config.retrieval.model_copy(
        update={"signal_mode": "subquestion_chain"}
    ).effective_for_dataset("musique").effective_for_signal_mode()
    assert sc.hierarchical_refinement_enabled is True
    assert sc.rerank_limit == 20
    assert sc.final_rerank_limit == 40
    assert sc.rerank_budget == 120
    assert sc.evidence_derived_state_anchors_enabled is False


def test_reranker_config_is_explicit_and_validated(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    write_json(base, base_payload())
    resolved = resolve_config(
        base_path=base,
        repo_root=tmp_path,
        environment={},
        cli_entries=(
            "reranker.type=qwen3",
            "reranker.device=cuda",
            "reranker.batch_size=2",
            "reranker.max_length=512",
        ),
    )

    assert resolved.config.reranker.type == "qwen3"
    assert resolved.config.reranker.device == "cuda"
    assert resolved.config.reranker.batch_size == 2
    assert resolved.config.reranker.max_length == 512


def test_recursive_secret_assertion() -> None:
    with pytest.raises(ConfigResolutionError):
        assert_no_secret_fields({"nested": {"access_token": "x"}})
    assert_no_secret_fields({"nested": {"api_key_env": "OPENAI_API_KEY"}})
