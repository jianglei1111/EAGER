from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

from eager.indexing.errors import IndexValidationError


def load_evidence_texts(
    path: Path,
    *,
    expected_ids: Collection[str] | None = None,
) -> dict[str, str]:
    """Load a strict evidence-id to alternate-text JSONL representation."""

    path = Path(path)
    output: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IndexValidationError(
                        f"evidence text representation cannot contain blank lines: "
                        f"{path}:{line_number}"
                    )
                payload = json.loads(line)
                if not isinstance(payload, dict) or set(payload) != {"evidence_id", "text"}:
                    raise IndexValidationError(
                        f"invalid evidence text record: {path}:{line_number}"
                    )
                evidence_id = payload["evidence_id"]
                text = payload["text"]
                if (
                    not isinstance(evidence_id, str)
                    or not evidence_id
                    or any(character.isspace() for character in evidence_id)
                    or not isinstance(text, str)
                    or not text.strip()
                ):
                    raise IndexValidationError(
                        f"invalid evidence text value: {path}:{line_number}"
                    )
                if evidence_id in output:
                    raise IndexValidationError(
                        f"duplicate evidence text ID: {path}:{line_number}"
                    )
                output[evidence_id] = text.strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexValidationError(f"cannot read evidence texts {path}: {exc}") from exc
    if expected_ids is not None:
        expected = set(expected_ids)
        actual = set(output)
        if actual != expected:
            raise IndexValidationError(
                "evidence text IDs do not match the canonical corpus: "
                f"missing={len(expected - actual)}, extra={len(actual - expected)}"
            )
    return output


__all__ = ["load_evidence_texts"]
