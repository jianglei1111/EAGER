from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence


_DATASET_PREFIXES = {
    "hotpotqa": "hp",
    "2wiki": "tw",
    "musique": "mu",
}


def normalize_content_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split())


def _dataset_prefix(dataset: str) -> str:
    normalized = dataset.strip().lower()
    known = _DATASET_PREFIXES.get(normalized)
    if known is not None:
        return known
    fallback = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return fallback or "data"


def make_paragraph_id(*, dataset: str, title: str, content: str | Sequence[str]) -> str:
    """Build a source-enumeration-independent identity from ordered paragraph content."""

    segments = [content] if isinstance(content, str) else list(content)
    payload = json.dumps(
        {
            "dataset": dataset.strip().lower(),
            "title": normalize_content_text(title),
            "segments": [normalize_content_text(segment) for segment in segments],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:32]
    return f"{_dataset_prefix(dataset)}_par_{digest}"


def make_sentence_id(paragraph_id: str, position: int) -> str:
    if position < 0:
        raise ValueError("sentence position must be non-negative")
    return f"{paragraph_id}_s{position}"


def encode_relation(parts: Sequence[object]) -> str:
    if len(parts) != 3:
        raise ValueError("a relation must contain subject, predicate, and object")
    if any(not isinstance(part, str) for part in parts):
        raise ValueError("relation fields must be text")
    cleaned = [normalize_content_text(part) for part in parts]
    if any(not part for part in cleaned):
        raise ValueError("relation fields must be non-empty")
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "encode_relation",
    "make_paragraph_id",
    "make_sentence_id",
    "normalize_content_text",
]
