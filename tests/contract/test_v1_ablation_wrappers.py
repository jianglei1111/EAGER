from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).parents[2]
SANDBOX = ROOT / "experiments_sandbox"

BASELINES = {
    ("hotpotqa", "rq"): (
        "retrieval_question",
        "hotpotqa_main_1000",
        "eager_v1_20260728_hotpotqa_rq_main1000_v1",
    ),
    ("hotpotqa", "sc"): (
        "subquestion_chain",
        "hotpotqa_main_1000",
        "eager_v1_20260728_hotpotqa_sc_main1000_v1",
    ),
    ("2wiki", "rq"): (
        "retrieval_question",
        "2wiki_main_1000",
        "eager_v1_20260728_2wiki_rq_main1000_v1",
    ),
    ("2wiki", "sc"): (
        "subquestion_chain",
        "2wiki_main_1000",
        "eager_v1_20260728_2wiki_sc_main1000_v1",
    ),
    ("musique", "rq"): (
        "retrieval_question",
        "musique_main_1000",
        "eager_v1_20260728_musique_rq_main1000_v1",
    ),
    ("musique", "sc"): (
        "subquestion_chain",
        "musique_main_1000",
        "eager_v1_20260728_musique_sc_main1000_v1",
    ),
}

ABLATIONS = {
    "without_entity_linked_activation": (
        "retrieval.entity_expansion_enabled=false",
        "entity_expansion_enabled",
        [True, False],
        tuple(BASELINES),
    ),
    "without_concurrent_semantic_lane": (
        "retrieval.candidate_union_enabled=false",
        "candidate_union_enabled",
        [True, False],
        tuple(BASELINES),
    ),
    "without_hierarchical_paragraph_gate": (
        "retrieval.hierarchical_refinement_enabled=false",
        "hierarchical_refinement_enabled",
        [True, False],
        tuple(BASELINES),
    ),
    "without_direct_state_anchors": (
        "retrieval.evidence_derived_state_anchors_enabled=false",
        "evidence_derived_state_anchors_enabled",
        [True, False],
        (("hotpotqa", "rq"), ("2wiki", "rq")),
    ),
}


def expected_matrix() -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    number = 33
    for ablation_name, (
        override,
        field,
        delta,
        dataset_modes,
    ) in ABLATIONS.items():
        for dataset, mode in dataset_modes:
            signal, split, baseline_id = BASELINES[(dataset, mode)]
            directory = (
                f"20260729_{number:03d}_{dataset}_{mode}_"
                f"{ablation_name}_main1000"
            )
            expected[directory] = {
                "dataset": dataset,
                "mode": mode,
                "signal": signal,
                "split": split,
                "baseline_id": baseline_id,
                "ablation_name": ablation_name,
                "override": override,
                "field": field,
                "delta": delta,
            }
            number += 1
    assert number == 53
    return expected


EXPECTED = expected_matrix()


def discover_wrappers() -> dict[str, Path]:
    return {
        directory.name: directory / "run.py"
        for directory in SANDBOX.iterdir()
        if directory.is_dir()
        and directory.name.startswith("20260729_")
        and 33 <= int(directory.name.split("_", 2)[1]) <= 52
        and (directory / "run.py").is_file()
    }


def load_wrapper(path: Path) -> ModuleType:
    module_name = f"_ablation_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prerequisites_exist(module: ModuleType) -> bool:
    required = (
        module.BASELINE_PREFLIGHT_PATH,
        module.BASELINE_PREDICTIONS_PATH,
        module.BASELINE_RESOLVED_PATH,
        module.PREPARED_DIR / "prepare_manifest.json",
        module.INDEX_DIR / "semantic" / "summary.json",
    )
    return all(path.is_file() for path in required)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def test_v1_component_ablation_matrix_is_exact() -> None:
    wrappers = discover_wrappers()
    assert set(wrappers) == set(EXPECTED)
    assert len(wrappers) == 20

    experiment_ids: set[str] = set()
    output_dirs: set[Path] = set()
    for directory, expectation in EXPECTED.items():
        module = load_wrapper(wrappers[directory])
        assert module.DATASET == expectation["dataset"]
        assert module.SPLIT == expectation["split"]
        assert module.SIGNAL_MODE == expectation["signal"]
        assert module.QUESTION_COUNT == 1000
        assert module.QUESTION_WORKERS == (6 if expectation["mode"] == "rq" else 10)
        assert module.BASELINE_EXPERIMENT_ID == expectation["baseline_id"]
        assert module.ABLATION_NAME == expectation["ablation_name"]
        assert module.OVERRIDES == (expectation["override"],)
        assert module.EXPECTED_CONFIG_DELTA == {
            expectation["field"]: expectation["delta"]
        }

        cli_args = module.build_cli_args()
        cli_overrides = [
            cli_args[index + 1]
            for index, value in enumerate(cli_args[:-1])
            if value == "--set"
        ]
        assert cli_overrides == [
            f"retrieval.signal_mode={expectation['signal']}",
            expectation["override"],
        ]
        experiment_ids.add(module.EXPERIMENT_ID)
        output_dirs.add(module.OUTPUT_DIR)

    assert len(experiment_ids) == 20
    assert len(output_dirs) == 20


def test_state_anchor_ablation_has_no_noop_configurations() -> None:
    wrappers = discover_wrappers()
    state_modules = [
        load_wrapper(path)
        for directory, path in wrappers.items()
        if "without_direct_state_anchors" in directory
    ]
    assert {(module.DATASET, module.SIGNAL_MODE) for module in state_modules} == {
        ("hotpotqa", "retrieval_question"),
        ("2wiki", "retrieval_question"),
    }


def test_real_preflights_reproduce_v1_and_change_only_declared_switch() -> None:
    wrappers = discover_wrappers()
    modules = [load_wrapper(path) for path in wrappers.values()]
    if not all(prerequisites_exist(module) for module in modules):
        pytest.skip("local V1 prepared data and paired artifacts are not installed")

    for module in modules:
        preflight = module.build_preflight()
        assert preflight["question_count"] == 1000
        assert preflight["annotation_visibility"] == "evaluation_only"
        assert preflight["prepared"]["inference_fields"] == ["dataset", "question"]
        assert preflight["paired_baseline"]["shared_source_file_count"] > 0
        assert set(preflight["index_artifacts"]["semantic"]["artifacts"]) == {
            "vectors",
            "evidence_ids",
            "ann",
        }
        assert len(preflight["index_artifacts"]["surface"]["artifacts"]) == 7
        assert set(preflight["paired_baseline"]["model_pins"]) == {
            "semantic_encoder",
            "reranker",
        }
        assert (
            preflight["paired_baseline"]["handle_id_sha256"]
            == preflight["prepared"]["handle_id_sha256"]
        )
        assert (
            preflight["ablation"]["actual_config_delta"]
            == module.EXPECTED_CONFIG_DELTA
        )


def test_check_only_is_read_only_and_does_not_enter_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_wrapper(
        SANDBOX
        / "20260729_033_hotpotqa_rq_without_entity_linked_activation_main1000"
        / "run.py"
    )
    if not prerequisites_exist(module):
        pytest.skip("local HotpotQA V1 artifacts are not installed")

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("check-only entered a mutating or model-execution path")

    output_dir = tmp_path / "must_not_be_created"
    monkeypatch.setattr(module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(module, "bind_preflight", forbidden)
    monkeypatch.setattr(module, "write_json", forbidden)
    monkeypatch.setattr(module, "execute_experiment", forbidden)
    monkeypatch.setattr(module, "require_clean_worktree", forbidden)

    assert module.main(["--check-only"]) == 0
    assert not output_dir.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "check_only_validated"
    assert payload["question_count"] == 1000


def test_preflight_binding_is_json_round_trip_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_wrapper(
        SANDBOX
        / "20260729_033_hotpotqa_rq_without_entity_linked_activation_main1000"
        / "run.py"
    )
    if not prerequisites_exist(module):
        pytest.skip("local HotpotQA V1 artifacts are not installed")

    preflight = module.build_preflight()
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "ablation_output")
    module.bind_preflight(preflight)
    module.bind_preflight(preflight)
    stored = module.read_json(module.OUTPUT_DIR / "canonical_preflight.json")
    assert stored["ablation"] == preflight["ablation"]


def test_formal_mode_checks_clean_worktree_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_wrapper(
        SANDBOX
        / "20260729_033_hotpotqa_rq_without_entity_linked_activation_main1000"
        / "run.py"
    )

    def dirty() -> None:
        raise RuntimeError("dirty worktree")

    def should_not_build() -> dict[str, Any]:
        raise AssertionError("preflight should not run after the dirty-worktree failure")

    monkeypatch.setattr(module, "require_clean_worktree", dirty)
    monkeypatch.setattr(module, "build_preflight", should_not_build)
    with pytest.raises(RuntimeError, match="dirty worktree"):
        module.main([])


def test_index_inventory_detects_semantic_and_surface_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_wrapper(
        SANDBOX
        / "20260729_033_hotpotqa_rq_without_entity_linked_activation_main1000"
        / "run.py"
    )
    index_dir = tmp_path / "index"
    semantic_dir = index_dir / "semantic"
    surface_dir = index_dir / "surface"
    semantic_dir.mkdir(parents=True)
    surface_dir.mkdir(parents=True)
    corpus_sha = "a" * 64

    semantic_entries: dict[str, dict[str, Any]] = {}
    semantic_names = {
        "vectors": "embeddings.npy",
        "evidence_ids": "evidence_ids.jsonl",
        "ann": "faiss_hnsw.index",
    }
    for key, name in semantic_names.items():
        path = semantic_dir / name
        path.write_bytes(f"{key}-artifact".encode())
        semantic_entries[key] = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": module.sha256_file(path),
        }
    write_json(
        semantic_dir / "summary.json",
        {"corpus": {"input_sha256": corpus_sha}, **semantic_entries},
    )
    write_json(surface_dir / "index_summary.json", {"schema_version": 1})
    for name in module.SURFACE_ARTIFACT_NAMES:
        (surface_dir / name).write_bytes(f"surface:{name}".encode())

    monkeypatch.setattr(module, "INDEX_DIR", index_dir)
    first = module.validate_index_artifacts(corpus_sha)
    output_dir = tmp_path / "bound_output"
    monkeypatch.setattr(module, "OUTPUT_DIR", output_dir)
    module.bind_preflight({"index_artifacts": first})

    surface_path = surface_dir / module.SURFACE_ARTIFACT_NAMES[0]
    surface_path.write_bytes(b"mutated-surface-posting")
    second = module.validate_index_artifacts(corpus_sha)
    with pytest.raises(ValueError, match="index_artifacts"):
        module.bind_preflight({"index_artifacts": second})

    semantic_path = semantic_dir / semantic_names["vectors"]
    semantic_path.write_bytes(b"mutated-semantic-vector")
    with pytest.raises(ValueError, match="semantic vectors artifact"):
        module.validate_index_artifacts(corpus_sha)


def test_completion_validation_binds_counts_ids_config_and_gptacc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_wrapper(
        SANDBOX
        / "20260729_033_hotpotqa_rq_without_entity_linked_activation_main1000"
        / "run.py"
    )
    prepared_dir = tmp_path / "prepared"
    output_dir = tmp_path / "output"
    ids = ["q1", "q2"]
    write_jsonl(
        prepared_dir / "handles.jsonl",
        [{"handle_id": handle_id} for handle_id in ids],
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "dataset": module.DATASET,
            "split": module.SPLIT,
            "question_count": 2,
            "question_workers": 2,
        },
    )
    aggregate = {
        "question_count": 2,
        "gpt_accuracy_question_count": 2,
        "successful_question_count": 2,
        "containment_accuracy": 0.5,
        "gpt_accuracy": 0.5,
        "error_question_count": 0,
        "error_rate": 0.0,
        "error_codes": {},
        "candidate_full_support_recall": 0.5,
        "paragraph_full_support_recall": 0.5,
        "llm_calls": 4,
        "reranker_pair_count": 10,
    }
    write_json(
        output_dir / "metrics.json",
        {
            "aggregate": aggregate,
            "questions": [
                {"handle_id": "q1", "gpt_accuracy": True, "status": "ok"},
                {"handle_id": "q2", "gpt_accuracy": False, "status": "ok"},
            ],
        },
    )
    effective = {"entity_expansion_enabled": False}
    baseline_resolved = output_dir / "baseline_resolved_config.json"
    write_json(
        baseline_resolved,
        {
            "experiment": {
                "effective_retrieval": {"entity_expansion_enabled": True},
                "question_workers": 8,
            }
        },
    )
    write_json(
        output_dir / "resolved_config.json",
        {"experiment": {"effective_retrieval": effective, "question_workers": 2}},
    )
    write_jsonl(
        output_dir / "predictions.jsonl",
        [
            {"handle_id": handle_id, "answer": "x", "status": "ok"}
            for handle_id in ids
        ],
    )
    write_jsonl(
        output_dir / "retrieval_traces.jsonl",
        [{"handle_id": handle_id, "run": {}} for handle_id in ids],
    )

    monkeypatch.setattr(module, "PREPARED_DIR", prepared_dir)
    monkeypatch.setattr(module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(module, "BASELINE_RESOLVED_PATH", baseline_resolved)
    monkeypatch.setattr(module, "QUESTION_COUNT", 2)
    monkeypatch.setattr(module, "QUESTION_WORKERS", 2)
    index_inventory = {"fixture": "bound"}
    monkeypatch.setattr(
        module,
        "validate_index_artifacts",
        lambda _corpus_sha: index_inventory,
    )
    baseline_contract = module.experiment_contract(module.read_json(baseline_resolved))
    preflight = {
        "effective_retrieval": effective,
        "paired_baseline": {
            "resolved_config_sha256": module.sha256_file(baseline_resolved),
            "experiment_contract_sha256": module.sha256_json(baseline_contract),
        },
        "prepared": {"corpus_sha256": "fixture-corpus"},
        "index_artifacts": index_inventory,
    }
    validation = module.validate_completed_run(preflight)
    assert validation["status"] == "validated"
    assert validation["question_count"] == 2

    write_json(
        output_dir / "resolved_config.json",
        {
            "experiment": {
                "effective_retrieval": effective,
                "question_workers": 2,
                "answer_prompt": "changed_prompt.txt",
            }
        },
    )
    with pytest.raises(RuntimeError, match="non-retrieval contract"):
        module.validate_completed_run(preflight)
    write_json(
        output_dir / "resolved_config.json",
        {"experiment": {"effective_retrieval": effective, "question_workers": 2}},
    )

    write_jsonl(
        output_dir / "retrieval_traces.jsonl",
        [{"handle_id": "q2", "run": {}}, {"handle_id": "q1", "run": {}}],
    )
    with pytest.raises(RuntimeError, match="misordered"):
        module.validate_completed_run(preflight)

    write_jsonl(
        output_dir / "retrieval_traces.jsonl",
        [{"handle_id": handle_id, "run": {}} for handle_id in ids],
    )
    aggregate.update(
        {
            "successful_question_count": 0,
            "error_question_count": 2,
            "error_rate": 1.0,
            "error_codes": {"question_execution_error": 2},
        }
    )
    write_json(
        output_dir / "metrics.json",
        {
            "aggregate": aggregate,
            "questions": [
                {"handle_id": handle_id, "gpt_accuracy": False, "status": "error"}
                for handle_id in ids
            ],
        },
    )
    write_jsonl(
        output_dir / "predictions.jsonl",
        [
            {"handle_id": handle_id, "answer": "", "status": "error"}
            for handle_id in ids
        ],
    )
    with pytest.raises(RuntimeError, match="catastrophic question error rate"):
        module.validate_completed_run(preflight)
