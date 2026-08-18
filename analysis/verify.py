from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "EXPORT_MANIFEST.json").is_file():
            return candidate
    raise FileNotFoundError("EXPORT_MANIFEST.json")


ROOT = _root()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def _csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_inventory(checks: list[str]) -> None:
    manifest = _json("EXPORT_MANIFEST.json")
    files = manifest.get("files")
    if not isinstance(files, list) or int(manifest.get("file_count", -1)) != len(files):
        raise ValueError("invalid export inventory")
    for item in files:
        path = ROOT / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
            raise ValueError(f"inventory mismatch: {item['path']}")

    forbidden_prefixes = (".git/", "external/", "paper/", "vendor/")
    exported = {str(item["path"]).replace("\\", "/") for item in files}
    if any(path.startswith(forbidden_prefixes) for path in exported):
        raise ValueError("non-EAGER source tree present in export")
    compatibility = {path for path in exported if path.startswith("src/eager_baselines/")}
    if compatibility != {"src/eager_baselines/__init__.py"}:
        raise ValueError("unexpected baseline implementation in EAGER-only export")
    checks.append(f"inventory: {len(files)} files, EAGER-only source boundary")


def verify_execution_snapshot(checks: list[str]) -> None:
    root = ROOT / "provenance" / "eager_v1_execution_snapshot"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    checkpoint = manifest["checkpoint_export_manifest"]
    if _sha256(root / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("checkpoint export manifest mismatch")

    for item in manifest["root_overlays"]:
        if _sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"result-bound root source mismatch: {item['path']}")
    for relative, expected in manifest["executed_main_wrappers"].items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"main wrapper mismatch: {relative}")
    for relative, expected in manifest["component_ablation_wrappers"].items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"ablation wrapper mismatch: {relative}")
    checks.append("execution snapshot: 6 main and 20 historical ablation wrappers bound")


def _read_ids(relative: str) -> list[str]:
    raw = (ROOT / relative).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError(f"non-canonical ID file: {relative}")
    values = raw.decode("utf-8").splitlines()
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate IDs: {relative}")
    return values


def verify_splits(checks: list[str]) -> None:
    groups = {
        "hotpotqa": (
            ("splits/formal/hotpotqa_bridge100.ids.txt", 100),
            ("splits/formal/hotpotqa_comparison100.ids.txt", 100),
            ("splits/formal/hotpotqa_remaining800.ids.txt", 800),
        ),
        "2wiki": (("splits/formal/2wiki_main1000.ids.txt", 1000),),
        "musique": (("splits/formal/musique_main1000.ids.txt", 1000),),
    }
    for dataset, specs in groups.items():
        combined: list[str] = []
        for relative, expected in specs:
            values = _read_ids(relative)
            if len(values) != expected:
                raise ValueError(f"ID count mismatch: {relative}")
            combined.extend(values)
        if len(combined) != 1000 or len(combined) != len(set(combined)):
            raise ValueError(f"invalid evaluation cohort: {dataset}")
    checks.append("splits: three 1,000-question cohorts are unique")


def verify_result_tables(checks: list[str]) -> None:
    expected_rows = {
        "results/p9_conacc_alias_f1_v1/system_metrics.csv": 21,
        "results/p9_conacc_alias_f1_v1/main_matched.csv": 6,
        "results/p9_conacc_alias_f1_v1/common_reader.csv": 12,
        "results/p9_conacc_alias_f1_v1/component_interventions.csv": 21,
        "results/p9_conacc_alias_f1_v1/repeated_question.csv": 3,
        "results/p9_conacc_alias_f1_v1/last_step_only.csv": 6,
        "results/p9_conacc_alias_f1_v1/structure.csv": 14,
        "results/p9_conacc_alias_f1_v1/hard200.csv": 3,
        "results/eager_v1_main1000_20260728/main_results.csv": 6,
        "results/standalone_signal_controls_v2_20260816/paired_eager_vs_standalone.csv": 6,
        "results/eager_v1_component_ablation_20260730/paired_deltas.csv": 20,
        "results/eager_anchor_structure_analysis_20260817/anchor_source_first_hit.csv": 6,
    }
    for relative, count in expected_rows.items():
        rows = _csv(relative)
        if len(rows) != count:
            raise ValueError(f"unexpected result row count: {relative}")

    figure_rows = _csv("results/scaling/figure2_scaling_cost_v1.csv")
    if len(figure_rows) != 63:
        raise ValueError("Figure 2 table must contain 63 rows")
    panels = Counter(row["panel"] for row in figure_rows)
    if panels != Counter(
        {"a_candidates": 18, "b_one_step_latency": 18, "c_full_rq_latency": 18, "d_quality_cost": 9}
    ):
        raise ValueError("Figure 2 panel counts differ")

    current_metrics = _csv("results/p9_conacc_alias_f1_v1/system_metrics.csv")
    required_columns = {"conacc_pct", "alias_f1_pct"}
    if not required_columns.issubset(current_metrics[0]):
        raise ValueError("current paper metrics must contain ConAcc and alias-aware F1")
    extension = _json("results/eager_v1_component_ablation_holm21_20260818/holm21_summary.json")
    if int(extension.get("family_size", -1)) != 21:
        raise ValueError("component intervention family must contain 21 comparisons")
    intervention = extension.get("new_intervention", {})
    if intervention.get("label") != "state_anchor_toggle:musique:rq":
        raise ValueError("MuSiQue-RQ state-anchor intervention is missing")
    if intervention.get("direction") != "enabled minus disabled":
        raise ValueError("state-anchor intervention direction is invalid")
    compact = _json("results/eager_v1_component_ablation_holm21_20260818/state_anchor_comparison.json")
    if int(compact.get("question_count", -1)) != 1000:
        raise ValueError("state-anchor comparison must contain 1,000 aligned questions")
    if compact.get("intervention") != "enabled minus disabled; all other frozen V1 settings unchanged":
        raise ValueError("state-anchor comparison is not a matched intervention")
    checks.append("component interventions: 20 historical wrappers plus one matched MuSiQue-RQ state-anchor comparison")
    checks.append("results: current ConAcc/F1, matched, ablation, structure, and scaling tables")


def main() -> int:
    checks: list[str] = []
    try:
        verify_inventory(checks)
        verify_execution_snapshot(checks)
        verify_splits(checks)
        verify_result_tables(checks)
    except Exception as exc:
        print(json.dumps({"ok": False, "checks": checks, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"ok": True, "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
