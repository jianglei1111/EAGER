"""Formal latest-V1 component ablation on one frozen main_1000 selection.

Hypothesis: Disables entity-linked/local candidate activation while retaining the fixed Top-20 semantic lane.
Control: eager_v1_20260728_2wiki_rq_main1000_v1
Changed variable: retrieval.entity_expansion_enabled=false
All other effective retrieval settings, prompts, prepared records, indexes,
question order, answer generator, reranker, and judge remain bound to V1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eager.config import resolve_config  # noqa: E402


DATASET = "2wiki"
SPLIT = "2wiki_main_1000"
SIGNAL_MODE = "retrieval_question"
QUESTION_COUNT = 1000
QUESTION_WORKERS = 6
MAX_INFRASTRUCTURE_ERROR_RATE = 0.05
BASELINE_EXPERIMENT_ID = "eager_v1_20260728_2wiki_rq_main1000_v1"
BASELINE_WRAPPER_DIR = "20260728_029_2wiki_rq_v1_full1000"
ABLATION_NAME = "without_entity_linked_activation"
ABLATION_DESCRIPTION = "Disables entity-linked/local candidate activation while retaining the fixed Top-20 semantic lane."
OVERRIDES = ("retrieval.entity_expansion_enabled=false",)
EXPECTED_CONFIG_DELTA = {
    "entity_expansion_enabled": [True, False],
}
EXPERIMENT_ID = (
    "eager_v1_20260729_2wiki_rq_without_entity_linked_activation_main1000_v1"
)

PREPARED_DIR = ROOT / "data" / "prepared" / "wsdm_v1" / SPLIT
INDEX_DIR = ROOT / "data" / "indexes" / "wsdm_v1" / SPLIT
OUTPUT_DIR = ROOT / "artifacts" / "test_runs" / "wsdm" / EXPERIMENT_ID
BASELINE_OUTPUT_DIR = (
    ROOT / "artifacts" / "test_runs" / "wsdm" / BASELINE_EXPERIMENT_ID
)
BASELINE_PREFLIGHT_PATH = BASELINE_OUTPUT_DIR / "canonical_preflight.json"
BASELINE_PREDICTIONS_PATH = BASELINE_OUTPUT_DIR / "predictions.jsonl"
BASELINE_RESOLVED_PATH = BASELINE_OUTPUT_DIR / "resolved_config.json"
BASELINE_WRAPPER_PATH = (
    ROOT / "experiments_sandbox" / BASELINE_WRAPPER_DIR / "run.py"
)
BASE_CONFIG = ROOT / "configs" / "wsdm" / "base.json"
LOCAL_CONFIG = ROOT / "configs" / "llm_config.local.json"
MODEL_REGISTRY = ROOT / "configs" / "wsdm" / "models" / "registry.json"
SEMANTIC_ARTIFACT_KEYS = ("vectors", "evidence_ids", "ann")
SURFACE_ARTIFACT_NAMES = (
    "entity_surface_index.json",
    "entity_to_sentence_map.json",
    "paragraph_to_window_map.json",
    "sentence_to_window_map.json",
    "title_surface_index.json",
    "token_surface_index.json",
    "window_store.jsonl",
)

PREPARED_PATHS = (
    PREPARED_DIR / "prepare_manifest.json",
    PREPARED_DIR / "corpus.jsonl",
    PREPARED_DIR / "corpus_summary.json",
    PREPARED_DIR / "inference.jsonl",
    PREPARED_DIR / "handles.jsonl",
    PREPARED_DIR / "annotations.jsonl",
)
BOUND_PATHS = (
    Path(__file__).resolve(),
    BASELINE_WRAPPER_PATH,
    BASE_CONFIG,
    LOCAL_CONFIG,
    MODEL_REGISTRY,
    ROOT / "prompts" / "eager_retrieval_question_system_prompt.md",
    ROOT / "prompts" / "eager_subquestion_chain_v6_simple_system_prompt.md",
    ROOT / "prompts" / "eager_intermediate_answer_system_prompt.md",
    ROOT / "prompts" / "linear_rag_answer_target_slot_system_prompt.txt",
    ROOT / "prompts" / "gpt_judge_prompt.txt",
    *PREPARED_PATHS,
    INDEX_DIR / "semantic" / "summary.json",
    INDEX_DIR / "surface" / "index_summary.json",
    ROOT / "src" / "eager" / "cli.py",
    ROOT / "src" / "eager" / "config.py",
    ROOT / "src" / "eager" / "schemas.py",
    ROOT / "src" / "eager" / "pipeline" / "engine.py",
    ROOT / "src" / "eager" / "retrieval" / "kernel.py",
    ROOT / "src" / "eager" / "retrieval" / "anchors.py",
    ROOT / "src" / "eager" / "retrieval" / "candidates.py",
    ROOT / "src" / "eager" / "retrieval" / "coarse_ranker.py",
    ROOT / "src" / "eager" / "retrieval" / "contracts.py",
    ROOT / "src" / "eager" / "retrieval" / "entity_mapping.py",
    ROOT / "src" / "eager" / "retrieval" / "hierarchical_refiner.py",
    ROOT / "src" / "eager" / "retrieval" / "reranker.py",
    ROOT / "src" / "eager" / "indexing" / "surface_store.py",
    ROOT / "src" / "eager" / "signals" / "retrieval_question.py",
    ROOT / "src" / "eager" / "signals" / "subquestion_chain.py",
    ROOT / "src" / "eager" / "answering" / "openai.py",
    ROOT / "src" / "eager" / "evaluation" / "judge.py",
    ROOT / "src" / "eager" / "experiments" / "runner.py",
)
FORBIDDEN_INFERENCE_KEYS = {
    "answer",
    "answer_aliases",
    "gold_relations",
    "gold_type",
    "hop_count",
    "official_decomposition",
    "supporting_paragraph_ids",
    "supporting_sentence_ids",
    "supporting_titles",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(handle_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(handle_ids) + "\n").encode("utf-8")
    ).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--short")),
    }


def require_clean_worktree() -> None:
    metadata = git_metadata()
    if metadata["dirty"]:
        raise RuntimeError(
            "formal ablation execution requires a clean Git worktree; "
            "commit the frozen wrappers before running"
        )


def effective_retrieval(cli_entries: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = resolve_config(
        base_path=BASE_CONFIG,
        local_path=LOCAL_CONFIG,
        repo_root=ROOT,
        environment=dict(os.environ),
        cli_entries=tuple(cli_entries),
    )
    effective = (
        resolved.config.retrieval.effective_for_dataset(DATASET)
        .effective_for_signal_mode()
    )
    return (
        effective.model_dump(mode="json", exclude={"dataset_profiles"}),
        resolved.public_dict()["llm"],
    )


def config_delta(
    control: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, list[Any]]:
    return {
        key: [control.get(key), variant.get(key)]
        for key in sorted(set(control) | set(variant))
        if control.get(key) != variant.get(key)
    }



def validate_inventory_entry(
    directory: Path,
    entry: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} inventory is not an object")
    relative = entry.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).name != relative
    ):
        raise ValueError(f"{label} inventory path is invalid")
    path = directory / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = entry.get("bytes")
    expected_sha = entry.get("sha256")
    if (
        not isinstance(expected_bytes, int)
        or path.stat().st_size != expected_bytes
        or not isinstance(expected_sha, str)
        or sha256_file(path) != expected_sha
    ):
        raise ValueError(f"{label} artifact differs from its inventory")
    return {
        "path": relative,
        "bytes": expected_bytes,
        "sha256": expected_sha,
    }


def validate_index_artifacts(corpus_sha: str) -> dict[str, Any]:
    semantic_dir = INDEX_DIR / "semantic"
    semantic = read_json(semantic_dir / "summary.json")
    if semantic.get("corpus", {}).get("input_sha256") != corpus_sha:
        raise ValueError("semantic index is not bound to the prepared corpus")
    expected_semantic_names = {
        "summary.json",
        *(
            str(semantic[key]["path"])
            for key in SEMANTIC_ARTIFACT_KEYS
            if isinstance(semantic.get(key), dict)
        ),
    }
    actual_semantic_names = {
        path.name for path in semantic_dir.iterdir() if path.is_file()
    }
    if actual_semantic_names != expected_semantic_names:
        raise ValueError("semantic index artifact set differs from its summary")
    semantic_inventory = {
        key: validate_inventory_entry(
            semantic_dir,
            semantic.get(key),
            label=f"semantic {key}",
        )
        for key in SEMANTIC_ARTIFACT_KEYS
    }

    surface_dir = INDEX_DIR / "surface"
    expected_surface_names = {"index_summary.json", *SURFACE_ARTIFACT_NAMES}
    actual_surface_names = {
        path.name for path in surface_dir.iterdir() if path.is_file()
    }
    if actual_surface_names != expected_surface_names:
        raise ValueError("surface index artifact set is incomplete or unexpected")
    surface_inventory = {
        name: {
            "bytes": (surface_dir / name).stat().st_size,
            "sha256": sha256_file(surface_dir / name),
        }
        for name in SURFACE_ARTIFACT_NAMES
    }
    return {
        "semantic": {
            "summary_sha256": sha256_file(semantic_dir / "summary.json"),
            "artifacts": semantic_inventory,
        },
        "surface": {
            "summary_sha256": sha256_file(surface_dir / "index_summary.json"),
            "artifacts": surface_inventory,
        },
    }


def experiment_contract(resolved: dict[str, Any]) -> dict[str, Any]:
    experiment = resolved.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("resolved configuration lacks experiment metadata")
    return {
        key: value
        for key, value in sorted(experiment.items())
        if key not in {"effective_retrieval", "question_workers"}
    }


def validate_model_registry(
    baseline_contract: dict[str, Any],
) -> dict[str, dict[str, str]]:
    registry = read_json(MODEL_REGISTRY)
    records = registry.get("models")
    if not isinstance(records, list):
        raise ValueError("model registry has no model records")
    by_name = {
        str(record.get("name")): record
        for record in records
        if isinstance(record, dict)
    }
    baseline_models = baseline_contract.get("models")
    if not isinstance(baseline_models, dict):
        raise ValueError("paired baseline lacks resolved model metadata")
    roles = {
        "semantic_encoder": "name",
        "reranker": "model_name",
    }
    pins: dict[str, dict[str, str]] = {}
    for role, name_field in roles.items():
        baseline = baseline_models.get(role)
        if not isinstance(baseline, dict):
            raise ValueError(f"paired baseline lacks {role} metadata")
        name = baseline.get(name_field)
        current = by_name.get(str(name))
        if current is None:
            raise ValueError(f"model registry lacks paired {role}: {name}")
        fields = (
            "model_id",
            "revision",
            "artifact_manifest_sha256",
            "pin_sha256",
        )
        mismatch = {
            field: (baseline.get(field), current.get(field))
            for field in fields
            if baseline.get(field) != current.get(field)
        }
        if mismatch:
            raise ValueError(f"{role} model pin differs from V1: {mismatch}")
        pins[role] = {field: str(current[field]) for field in fields}
    return pins


def validate_prepared_and_alignment() -> tuple[list[str], str, dict[str, Any]]:
    for path in (
        *BOUND_PATHS,
        BASELINE_PREFLIGHT_PATH,
        BASELINE_PREDICTIONS_PATH,
        BASELINE_RESOLVED_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    prepared = read_json(PREPARED_DIR / "prepare_manifest.json")
    if (
        prepared.get("dataset") != DATASET
        or prepared.get("split") != SPLIT
        or int(prepared.get("record_count", -1)) != QUESTION_COUNT
    ):
        raise ValueError("prepared dataset does not match the frozen V1 selection")

    output_names = {
        "annotations.jsonl",
        "corpus.jsonl",
        "corpus_summary.json",
        "handles.jsonl",
        "inference.jsonl",
    }
    outputs = prepared.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != output_names:
        raise ValueError("prepared manifest output inventory is incomplete")
    for name in sorted(output_names):
        entry = outputs.get(name)
        if not isinstance(entry, dict) or entry.get("path") != name:
            raise ValueError(f"invalid prepared manifest entry: {name}")
        path = PREPARED_DIR / name
        if (
            path.stat().st_size != int(entry.get("bytes", -1))
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError(f"prepared output differs from its manifest: {name}")

    corpus_sha = sha256_file(PREPARED_DIR / "corpus.jsonl")
    index_inventory = validate_index_artifacts(corpus_sha)

    inference_count = 0
    for row in iter_jsonl(PREPARED_DIR / "inference.jsonl"):
        inference_count += 1
        if set(row) != {"dataset", "question"}:
            unexpected = sorted(set(row) - {"dataset", "question"})
            gold = sorted(set(row) & FORBIDDEN_INFERENCE_KEYS)
            raise ValueError(
                f"inference row violates the gold-free schema: "
                f"unexpected={unexpected}, gold={gold}"
            )
        if row["dataset"] != DATASET or not str(row["question"]).strip():
            raise ValueError("invalid inference dataset or empty question")
    if inference_count != QUESTION_COUNT:
        raise ValueError("inference record count is not 1,000")

    handle_rows = list(iter_jsonl(PREPARED_DIR / "handles.jsonl"))
    handle_ids = [str(row["handle_id"]) for row in handle_rows]
    if (
        len(handle_ids) != QUESTION_COUNT
        or len(set(handle_ids)) != QUESTION_COUNT
        or any(row.get("dataset") != DATASET or row.get("split") != SPLIT for row in handle_rows)
    ):
        raise ValueError("prepared handle registry is invalid")

    annotation_ids = [
        str(row["handle_id"])
        for row in iter_jsonl(PREPARED_DIR / "annotations.jsonl")
    ]
    if annotation_ids != handle_ids:
        raise ValueError("evaluation annotations are not aligned with inference handles")

    baseline_ids = [
        str(row["handle_id"])
        for row in iter_jsonl(BASELINE_PREDICTIONS_PATH)
    ]
    if baseline_ids != handle_ids:
        raise ValueError("paired V1 predictions use a different question selection or order")
    return handle_ids, corpus_sha, index_inventory


def validate_baseline_binding(
    baseline_preflight: dict[str, Any],
    control_settings: dict[str, Any],
    current_llm: dict[str, Any],
) -> int:
    expected_identity = {
        "experiment_id": BASELINE_EXPERIMENT_ID,
        "dataset": DATASET,
        "split": SPLIT,
        "signal_mode": SIGNAL_MODE,
        "question_count": QUESTION_COUNT,
        "annotation_visibility": "evaluation_only",
    }
    mismatches = {
        key: (baseline_preflight.get(key), value)
        for key, value in expected_identity.items()
        if baseline_preflight.get(key) != value
    }
    if mismatches:
        raise ValueError(f"paired V1 preflight identity mismatch: {mismatches}")
    if baseline_preflight.get("effective_retrieval") != control_settings:
        raise ValueError("current full configuration does not reproduce the paired V1 control")
    if baseline_preflight.get("llm") != current_llm:
        raise ValueError("current public LLM settings differ from the paired V1 control")

    source_hashes = baseline_preflight.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("paired V1 preflight lacks source hashes")
    checked = 0
    for relative, expected_sha in source_hashes.items():
        if str(relative).startswith("experiments_sandbox/"):
            continue
        path = ROOT / str(relative)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"paired V1 source binding changed: {relative}")
        checked += 1
    if checked == 0:
        raise ValueError("no shared V1 source binding was validated")
    return checked


def build_preflight() -> dict[str, Any]:
    handle_ids, corpus_sha, index_inventory = validate_prepared_and_alignment()
    baseline_preflight = read_json(BASELINE_PREFLIGHT_PATH)
    baseline_resolved = read_json(BASELINE_RESOLVED_PATH)

    control_settings, control_llm = effective_retrieval(
        (f"retrieval.signal_mode={SIGNAL_MODE}",)
    )
    baseline_effective = baseline_resolved.get("experiment", {}).get(
        "effective_retrieval"
    )
    if baseline_effective != control_settings:
        raise ValueError("paired resolved V1 control differs from current full config")
    baseline_contract = experiment_contract(baseline_resolved)
    model_pins = validate_model_registry(baseline_contract)
    shared_source_count = validate_baseline_binding(
        baseline_preflight,
        control_settings,
        control_llm,
    )
    variant_settings, variant_llm = effective_retrieval(
        (f"retrieval.signal_mode={SIGNAL_MODE}", *OVERRIDES)
    )
    if variant_llm != control_llm:
        raise ValueError("the ablation unexpectedly changed public LLM settings")
    actual_delta = config_delta(control_settings, variant_settings)
    if actual_delta != EXPECTED_CONFIG_DELTA:
        raise ValueError(
            f"ablation config delta mismatch: "
            f"expected={EXPECTED_CONFIG_DELTA}, actual={actual_delta}"
        )

    return {
        "schema_version": 2,
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "split": SPLIT,
        "signal_mode": SIGNAL_MODE,
        "question_count": QUESTION_COUNT,
        "question_workers": QUESTION_WORKERS,
        "execution_profile": "wsdm_strict",
        "git": git_metadata(),
        "ablation": {
            "name": ABLATION_NAME,
            "description": ABLATION_DESCRIPTION,
            "overrides": list(OVERRIDES),
            "expected_config_delta": EXPECTED_CONFIG_DELTA,
            "actual_config_delta": actual_delta,
        },
        "paired_baseline": {
            "experiment_id": BASELINE_EXPERIMENT_ID,
            "preflight_sha256": sha256_file(BASELINE_PREFLIGHT_PATH),
            "resolved_config_sha256": sha256_file(BASELINE_RESOLVED_PATH),
            "experiment_contract_sha256": sha256_json(baseline_contract),
            "handle_id_sha256": sha256_ids(handle_ids),
            "shared_source_file_count": shared_source_count,
            "model_pins": model_pins,
        },
        "effective_retrieval": variant_settings,
        "llm": variant_llm,
        "prepared": {
            "corpus_sha256": corpus_sha,
            "handle_id_sha256": sha256_ids(handle_ids),
            "inference_fields": ["dataset", "question"],
            "annotation_visibility": "evaluation_only",
        },
        "index_artifacts": index_inventory,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in BOUND_PATHS
        },
        "annotation_visibility": "evaluation_only",
        "paper_metric_note": (
            "MuSiQue ConAcc must be recomputed canonical-only with "
            "wsdm/scripts/build_table1_canonical_fairness_v2.py."
            if DATASET == "musique"
            else "Runner ConAcc and canonical-only ConAcc use the same reference."
        ),
    }


def bind_preflight(preflight: dict[str, Any]) -> None:
    path = OUTPUT_DIR / "canonical_preflight.json"
    if not path.exists():
        write_json(path, preflight)
        return
    previous = read_json(path)
    stable_keys = (
        "schema_version",
        "experiment_id",
        "dataset",
        "split",
        "signal_mode",
        "question_count",
        "question_workers",
        "execution_profile",
        "git",
        "ablation",
        "paired_baseline",
        "effective_retrieval",
        "llm",
        "prepared",
        "index_artifacts",
        "source_sha256",
        "annotation_visibility",
        "paper_metric_note",
    )
    for key in stable_keys:
        if previous.get(key) != preflight.get(key):
            raise ValueError(f"existing canonical preflight differs at {key}")


def nested_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_keys(item)


def validate_completed_run(preflight: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(OUTPUT_DIR / "run_manifest.json")
    metrics = read_json(OUTPUT_DIR / "metrics.json")
    resolved = read_json(OUTPUT_DIR / "resolved_config.json")
    aggregate = metrics["aggregate"]

    if (
        manifest.get("status") != "complete"
        or manifest.get("dataset") != DATASET
        or manifest.get("split") != SPLIT
        or int(manifest.get("question_count", -1)) != QUESTION_COUNT
        or int(manifest.get("question_workers", -1)) != QUESTION_WORKERS
    ):
        raise RuntimeError("formal ablation run is not complete or is misbound")
    if (
        int(aggregate.get("question_count", -1)) != QUESTION_COUNT
        or int(aggregate.get("gpt_accuracy_question_count", -1)) != QUESTION_COUNT
    ):
        raise RuntimeError("metrics or GPTAcc decision count is incomplete")
    if resolved["experiment"]["effective_retrieval"] != preflight["effective_retrieval"]:
        raise RuntimeError("executed retrieval configuration differs from preflight")
    if int(resolved["experiment"].get("question_workers", -1)) != QUESTION_WORKERS:
        raise RuntimeError("executed question-worker count differs from preflight")
    if (
        sha256_file(BASELINE_RESOLVED_PATH)
        != preflight["paired_baseline"]["resolved_config_sha256"]
    ):
        raise RuntimeError("paired V1 resolved configuration changed after preflight")
    baseline_contract = experiment_contract(read_json(BASELINE_RESOLVED_PATH))
    if (
        sha256_json(baseline_contract)
        != preflight["paired_baseline"]["experiment_contract_sha256"]
        or experiment_contract(resolved) != baseline_contract
    ):
        raise RuntimeError("executed non-retrieval contract differs from paired V1")
    current_index_inventory = validate_index_artifacts(
        preflight["prepared"]["corpus_sha256"]
    )
    if current_index_inventory != preflight["index_artifacts"]:
        raise RuntimeError("index artifacts changed after canonical preflight")

    expected_ids = [
        str(row["handle_id"])
        for row in iter_jsonl(PREPARED_DIR / "handles.jsonl")
    ]
    prediction_rows = list(iter_jsonl(OUTPUT_DIR / "predictions.jsonl"))
    trace_rows = list(iter_jsonl(OUTPUT_DIR / "retrieval_traces.jsonl"))
    question_metrics = metrics.get("questions")
    if not isinstance(question_metrics, list):
        raise RuntimeError("metrics does not contain per-question GPTAcc decisions")

    for name, rows in (
        ("predictions", prediction_rows),
        ("retrieval traces", trace_rows),
        ("metric decisions", question_metrics),
    ):
        ids = [str(row.get("handle_id", "")) for row in rows]
        if ids != expected_ids:
            raise RuntimeError(f"{name} are incomplete, duplicated, or misordered")
    if any(not isinstance(row.get("gpt_accuracy"), bool) for row in question_metrics):
        raise RuntimeError("not every question has a Boolean GPTAcc decision")

    metric_error_ids = [
        str(row.get("handle_id", ""))
        for row in question_metrics
        if row.get("status") != "ok"
    ]
    prediction_error_ids = [
        str(row.get("handle_id", ""))
        for row in prediction_rows
        if row.get("status") != "ok"
    ]
    error_count = int(aggregate.get("error_question_count", -1))
    successful_count = int(aggregate.get("successful_question_count", -1))
    if (
        metric_error_ids != prediction_error_ids
        or len(metric_error_ids) != error_count
        or successful_count != QUESTION_COUNT - error_count
    ):
        raise RuntimeError("question error accounting is inconsistent")
    error_rate = error_count / QUESTION_COUNT
    infrastructure_error_count = sum(
        1
        for row in question_metrics
        if row.get("status") != "ok"
        and row.get("error_code") != "answer_format_error"
    )
    infrastructure_error_rate = infrastructure_error_count / QUESTION_COUNT
    if (
        error_count == QUESTION_COUNT
        or infrastructure_error_rate > MAX_INFRASTRUCTURE_ERROR_RATE
    ):
        raise RuntimeError(
            "catastrophic question error rate: "
            f"total={error_count}/{QUESTION_COUNT}, "
            f"infrastructure={infrastructure_error_count}/{QUESTION_COUNT}"
        )

    forbidden_output_keys = (FORBIDDEN_INFERENCE_KEYS - {"answer"}) | {
        "raw_dataset_id"
    }
    for name, rows in (("predictions", prediction_rows), ("retrieval traces", trace_rows)):
        leaked = forbidden_output_keys & {
            key for row in rows for key in nested_keys(row)
        }
        if leaked:
            raise RuntimeError(f"{name} contain evaluation-only metadata keys: {sorted(leaked)}")

    return {
        "status": "validated",
        "experiment_id": EXPERIMENT_ID,
        "paired_baseline_experiment_id": BASELINE_EXPERIMENT_ID,
        "ablation_name": ABLATION_NAME,
        "question_count": QUESTION_COUNT,
        "runner_containment_accuracy_alias_aware": aggregate["containment_accuracy"],
        "gpt_accuracy": aggregate["gpt_accuracy"],
        "gpt_judged_successful_question_count": successful_count,
        "error_question_count": error_count,
        "error_rate": error_rate,
        "infrastructure_error_question_count": infrastructure_error_count,
        "infrastructure_error_rate": infrastructure_error_rate,
        "error_codes": aggregate.get("error_codes", {}),
        "candidate_full_support_recall": aggregate["candidate_full_support_recall"],
        "final_full_support_recall": aggregate["paragraph_full_support_recall"],
        "llm_calls": aggregate["llm_calls"],
        "reranker_pair_count": aggregate["reranker_pair_count"],
        "canonical_only_recompute_required": DATASET == "musique",
        "annotation_visibility": "evaluation_only",
    }


def build_cli_args() -> list[str]:
    args = [
        "experiment",
        "run",
        "--base",
        str(BASE_CONFIG.relative_to(ROOT)),
        "--local",
        str(LOCAL_CONFIG.relative_to(ROOT)),
        "--set",
        f"retrieval.signal_mode={SIGNAL_MODE}",
    ]
    for override in OVERRIDES:
        args.extend(("--set", override))
    args.extend(
        (
            "--prepared",
            str(PREPARED_DIR.relative_to(ROOT)),
            "--index",
            str(INDEX_DIR.relative_to(ROOT)),
            "--run-dir",
            str(OUTPUT_DIR.relative_to(ROOT)),
            "--profile",
            "wsdm_strict",
            "--final-answer-scope",
            "global",
            "--semantic-device",
            "cuda",
            "--answer-prompt",
            "prompts/linear_rag_answer_target_slot_system_prompt.txt",
            "--retrieval-question-prompt",
            "prompts/eager_retrieval_question_system_prompt.md",
        )
    )
    if SIGNAL_MODE == "subquestion_chain":
        args.extend(
            (
                "--chain-prompt",
                "prompts/eager_subquestion_chain_v6_simple_system_prompt.md",
                "--intermediate-answer-prompt",
                "prompts/eager_intermediate_answer_system_prompt.md",
            )
        )
    args.extend(
        (
            "--judge",
            "--judge-prompt",
            "prompts/gpt_judge_prompt.txt",
            "--question-workers",
            str(QUESTION_WORKERS),
        )
    )
    return args


def execute_experiment() -> int:
    from eager.cli import main as eager_main

    return int(eager_main(build_cli_args()))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate all bindings and the exact config delta without model execution",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        preflight = build_preflight()
        print(
            json.dumps(
                {
                    "status": "check_only_validated",
                    "experiment_id": EXPERIMENT_ID,
                    "paired_baseline_experiment_id": BASELINE_EXPERIMENT_ID,
                    "ablation_name": ABLATION_NAME,
                    "question_count": QUESTION_COUNT,
                    "actual_config_delta": preflight["ablation"]["actual_config_delta"],
                    "handle_id_sha256": preflight["prepared"]["handle_id_sha256"],
                    "annotation_visibility": preflight["annotation_visibility"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    require_clean_worktree()
    preflight = build_preflight()
    bind_preflight(preflight)
    code = execute_experiment()
    if code != 0:
        return code
    validation = validate_completed_run(preflight)
    write_json(OUTPUT_DIR / "canonical_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
