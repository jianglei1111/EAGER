from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "experiments_sandbox"
        ).is_dir():
            return candidate
    raise FileNotFoundError("cannot locate the artifact root")


ROOT = _find_root()
PROVENANCE = ROOT / "provenance" / "eager_v1_execution_snapshot"
if not PROVENANCE.is_dir():
    PROVENANCE = ROOT / "wsdm" / "reproduction" / "eager_v1_execution_snapshot"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check or materialize the wrapper set bound to an EAGER V1 protocol."
    )
    parser.add_argument("--mode", choices=("main", "ablation"), required=True)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((PROVENANCE / "manifest.json").read_text(encoding="utf-8"))
    key = "executed_main_wrappers" if args.mode == "main" else "ablation_base_wrappers"
    source_dir = "main_wrappers_executed" if args.mode == "main" else "ablation_base_wrappers"
    results = []
    for relative, expected in manifest[key].items():
        archived = PROVENANCE / source_dir / relative
        if _sha256(archived) != expected:
            raise ValueError(f"archived wrapper hash mismatch: {relative}")
        target = ROOT / relative
        if args.materialize:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archived, target)
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"active wrapper does not match {args.mode}: {relative}")
        results.append(relative)

    print(
        json.dumps(
            {
                "ok": True,
                "mode": args.mode,
                "materialized": args.materialize,
                "wrapper_count": len(results),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
