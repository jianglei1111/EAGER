"""Canonical latest-V1 EAGER run for one frozen main_1000 configuration.

This wrapper does not alter retrieval behavior. It binds the current unified
CLI, configuration, prompts, prepared data, index metadata, and key source
files before starting one complete run with one shared Qwen reranker.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eager.cli import main as eager_main  # noqa: E402
from eager.config import resolve_config  # noqa: E402


DATASET = "2wiki"
SPLIT = "2wiki_main_1000"
SIGNAL_MODE = "subquestion_chain"
QUESTION_COUNT = 1000
QUESTION_WORKERS = 12
EXPERIMENT_ID = "eager_v1_20260728_2wiki_sc_main1000_v1"
EXPECTED_STATE_ANCHORS = False
EXPECTED_SCOPE = "latest_step"
EXPECTED_ORIGINAL_WEIGHT = 0.0
EXPECTED_SIGNAL_WEIGHT = 1.0
EXPECTED_MAX_ITERATIONS = 3
EXPECTED_CANDIDATE_BUDGET = 150
EXPECTED_PARAGRAPH_BUDGET = 15
EXPECTED_RERANK_BUDGET = 100

PREPARED_DIR = ROOT / "data" / "prepared" / "wsdm_v1" / SPLIT
INDEX_DIR = ROOT / "data" / "indexes" / "wsdm_v1" / SPLIT
OUTPUT_DIR = ROOT / "artifacts" / "test_runs" / "wsdm" / EXPERIMENT_ID
BASE_CONFIG = ROOT / "configs" / "wsdm" / "base.json"
LOCAL_CONFIG = ROOT / "configs" / "llm_config.local.json"

BOUND_PATHS = (
    Path(__file__).resolve(),
    BASE_CONFIG,
    LOCAL_CONFIG,
    ROOT / "prompts" / "eager_retrieval_question_system_prompt.md",
    ROOT / "prompts" / "eager_subquestion_chain_v6_simple_system_prompt.md",
    ROOT / "prompts" / "eager_intermediate_answer_system_prompt.md",
    ROOT / "prompts" / "linear_rag_answer_target_slot_system_prompt.txt",
    ROOT / "prompts" / "gpt_judge_prompt.txt",
    PREPARED_DIR / "prepare_manifest.json",
    PREPARED_DIR / "corpus.jsonl",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--short")),
    }


def build_preflight() -> dict[str, Any]:
    for path in BOUND_PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)
    prepared = read_json(PREPARED_DIR / "prepare_manifest.json")
    if (
        prepared.get("dataset") != DATASET
        or prepared.get("split") != SPLIT
        or int(prepared.get("record_count", -1)) != QUESTION_COUNT
    ):
        raise ValueError("prepared dataset does not match the canonical run")
    semantic = read_json(INDEX_DIR / "semantic" / "summary.json")
    if semantic["corpus"]["input_sha256"] != sha256_file(PREPARED_DIR / "corpus.jsonl"):
        raise ValueError("semantic index is not bound to the prepared corpus")

    resolved = resolve_config(
        base_path=BASE_CONFIG,
        local_path=LOCAL_CONFIG,
        repo_root=ROOT,
        environment=dict(os.environ),
        cli_entries=(f"retrieval.signal_mode={SIGNAL_MODE}",),
    )
    effective = (
        resolved.config.retrieval.effective_for_dataset(DATASET)
        .effective_for_signal_mode()
    )
    settings = effective.model_dump(mode="json", exclude={"dataset_profiles"})
    expected = {
        "signal_mode": SIGNAL_MODE,
        "evidence_derived_state_anchors_enabled": EXPECTED_STATE_ANCHORS,
        "hierarchical_refinement_enabled": True,
        "candidate_union_enabled": True,
        "candidate_union_semantic_limit": 20,
        "rerank_limit": 20,
        "final_rerank_limit": 40,
        "evidence_limit": 30,
        "original_query_weight": EXPECTED_ORIGINAL_WEIGHT,
        "signal_query_weight": EXPECTED_SIGNAL_WEIGHT,
        "retained_anchor_scope": EXPECTED_SCOPE,
        "rq_evidence_window_mode": "oldest",
        "max_iterations": EXPECTED_MAX_ITERATIONS,
        "candidate_budget": EXPECTED_CANDIDATE_BUDGET,
        "paragraph_budget": EXPECTED_PARAGRAPH_BUDGET,
        "rerank_budget": EXPECTED_RERANK_BUDGET,
    }
    changed = {
        key: (settings.get(key), value)
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if changed:
        raise ValueError(f"effective V1 configuration mismatch: {changed}")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "split": SPLIT,
        "signal_mode": SIGNAL_MODE,
        "question_count": QUESTION_COUNT,
        "question_workers": QUESTION_WORKERS,
        "execution_profile": "wsdm_strict",
        "git": git_metadata(),
        "effective_retrieval": settings,
        "llm": resolved.public_dict()["llm"],
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in BOUND_PATHS
        },
        "annotation_visibility": "evaluation_only",
    }


def bind_preflight(preflight: dict[str, Any]) -> None:
    path = OUTPUT_DIR / "canonical_preflight.json"
    if not path.exists():
        write_json(path, preflight)
        return
    previous = read_json(path)
    stable_keys = (
        "experiment_id",
        "dataset",
        "split",
        "signal_mode",
        "question_count",
        "question_workers",
        "execution_profile",
        "git",
        "effective_retrieval",
        "llm",
        "source_sha256",
        "annotation_visibility",
    )
    for key in stable_keys:
        if previous.get(key) != preflight.get(key):
            raise ValueError(f"existing canonical preflight differs at {key}")


def validate_completed_run(preflight: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(OUTPUT_DIR / "run_manifest.json")
    metrics = read_json(OUTPUT_DIR / "metrics.json")
    resolved = read_json(OUTPUT_DIR / "resolved_config.json")
    aggregate = metrics["aggregate"]
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("question_count", -1)) != QUESTION_COUNT
    ):
        raise RuntimeError("canonical run is not complete")
    if int(aggregate.get("question_count", -1)) != QUESTION_COUNT:
        raise RuntimeError("metrics question count is incomplete")
    if int(aggregate.get("gpt_accuracy_question_count", -1)) != QUESTION_COUNT:
        raise RuntimeError("GPTAcc judgment count is incomplete")
    if resolved["experiment"]["effective_retrieval"] != preflight["effective_retrieval"]:
        raise RuntimeError("executed retrieval configuration differs from preflight")
    for name in ("predictions.jsonl", "retrieval_traces.jsonl"):
        with (OUTPUT_DIR / name).open("r", encoding="utf-8") as handle:
            if sum(1 for line in handle if line.strip()) != QUESTION_COUNT:
                raise RuntimeError(f"{name} does not contain exactly 1,000 records")
    return {
        "status": "validated",
        "experiment_id": EXPERIMENT_ID,
        "question_count": QUESTION_COUNT,
        "containment_accuracy": aggregate["containment_accuracy"],
        "gpt_accuracy": aggregate["gpt_accuracy"],
        "error_question_count": aggregate["error_question_count"],
        "candidate_full_support_recall": aggregate["candidate_full_support_recall"],
        "final_full_support_recall": aggregate["paragraph_full_support_recall"],
        "llm_calls": aggregate["llm_calls"],
        "reranker_pair_count": aggregate["reranker_pair_count"],
    }


def main() -> int:
    preflight = build_preflight()
    bind_preflight(preflight)
    args = [
        "experiment", "run",
        "--base", str(BASE_CONFIG.relative_to(ROOT)),
        "--local", str(LOCAL_CONFIG.relative_to(ROOT)),
        "--set", f"retrieval.signal_mode={SIGNAL_MODE}",
        "--prepared", str(PREPARED_DIR.relative_to(ROOT)),
        "--index", str(INDEX_DIR.relative_to(ROOT)),
        "--run-dir", str(OUTPUT_DIR.relative_to(ROOT)),
        "--profile", "wsdm_strict",
        "--final-answer-scope", "global",
        "--semantic-device", "cuda",
        "--answer-prompt", "prompts/linear_rag_answer_target_slot_system_prompt.txt",
        "--retrieval-question-prompt", "prompts/eager_retrieval_question_system_prompt.md",
        "--chain-prompt", "prompts/eager_subquestion_chain_v6_simple_system_prompt.md",
        "--intermediate-answer-prompt", "prompts/eager_intermediate_answer_system_prompt.md",
        "--judge",
        "--judge-prompt", "prompts/gpt_judge_prompt.txt",
        "--question-workers", str(QUESTION_WORKERS),
    ]
    code = eager_main(args)
    if code != 0:
        return int(code)
    validation = validate_completed_run(preflight)
    write_json(OUTPUT_DIR / "canonical_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

