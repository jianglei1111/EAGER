from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from eager.indexing.corpus import CorpusRecord, CorpusSnapshot, load_corpus_snapshot
from eager.runtime.io import (
    atomic_write_json,
    atomic_write_jsonl,
    iter_jsonl,
    sha256_file,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "and",
        "are",
        "before",
        "did",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "into",
        "its",
        "that",
        "the",
        "their",
        "this",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whose",
        "with",
    }
)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) >= 3 and token not in _STOPWORDS
    )


def _record_evidence_count(record: CorpusRecord) -> int:
    return sum(bool(sentence) for sentence in record.sentences) if record.sentences else 1


def _hash_key(seed: int, paragraph_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{paragraph_id}".encode("utf-8")).hexdigest()


def _query_weights(inference_path: Path) -> dict[str, float]:
    frequencies: Counter[str] = Counter()
    count = 0
    for row in iter_jsonl(inference_path):
        if not isinstance(row, dict) or set(row) != {"dataset", "question"}:
            raise ValueError("scaling inference must contain only dataset and question")
        count += 1
        frequencies.update(_tokens(str(row["question"])))
    if count == 0:
        raise ValueError("scaling inference is empty")
    return {token: math.log1p(count / frequency) for token, frequency in frequencies.items()}


def _candidate_rows(
    *,
    base: CorpusSnapshot,
    distractors: Sequence[CorpusSnapshot],
    query_weights: dict[str, float],
    seed: int,
) -> list[dict[str, object]]:
    base_ids = {record.paragraph_id for record in base.records}
    by_id: dict[str, CorpusRecord] = {}
    for snapshot in distractors:
        for record in snapshot.records:
            if record.paragraph_id not in base_ids:
                by_id.setdefault(record.paragraph_id, record)
    rows: list[dict[str, object]] = []
    for record in by_id.values():
        title_tokens = _tokens(record.title)
        text_tokens = _tokens(record.text)
        score = sum(query_weights.get(token, 0.0) * 2.0 for token in title_tokens)
        score += sum(query_weights.get(token, 0.0) for token in text_tokens)
        rows.append(
            {
                "record": record,
                "paragraph_id": record.paragraph_id,
                "hard_score": round(score, 8),
                "random_key": _hash_key(seed, record.paragraph_id),
                "evidence_count": _record_evidence_count(record),
            }
        )
    return rows


def _mixed_order(rows: list[dict[str, object]], *, hard_fraction: float) -> list[dict[str, object]]:
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be between 0 and 1")
    hard = sorted(
        rows,
        key=lambda row: (
            -float(row["hard_score"]),
            str(row["random_key"]),
            str(row["paragraph_id"]),
        ),
    )
    random_rows = sorted(
        rows, key=lambda row: (str(row["random_key"]), str(row["paragraph_id"]))
    )
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    hard_index = 0
    random_index = 0
    hard_selected = 0

    def take(source: list[dict[str, object]], index: int) -> tuple[dict[str, object] | None, int]:
        while index < len(source):
            candidate = source[index]
            index += 1
            if str(candidate["paragraph_id"]) not in seen:
                return candidate, index
        return None, index

    while len(output) < len(rows):
        choose_hard = hard_selected < math.ceil((len(output) + 1) * hard_fraction)
        source = hard if choose_hard else random_rows
        index = hard_index if choose_hard else random_index
        candidate, index = take(source, index)
        if choose_hard:
            hard_index = index
        else:
            random_index = index
        if candidate is None:
            alternate = random_rows if choose_hard else hard
            alternate_index = random_index if choose_hard else hard_index
            candidate, alternate_index = take(alternate, alternate_index)
            if choose_hard:
                random_index = alternate_index
            else:
                hard_index = alternate_index
        if candidate is None:
            break
        paragraph_id = str(candidate["paragraph_id"])
        seen.add(paragraph_id)
        selected = dict(candidate)
        selected["selection_source"] = "hard" if choose_hard else "random"
        hard_selected += int(choose_hard)
        output.append(selected)
    return output


def build_scaling_corpora(
    *,
    base_corpus: Path,
    inference_path: Path,
    distractor_corpora: Sequence[Path],
    output_root: Path,
    scales: Sequence[int] = (1, 2, 4, 8),
    seed: int = 20260713,
    hard_fraction: float = 0.5,
    repo_root: Path | None = None,
) -> dict[str, object]:
    normalized_scales = tuple(sorted(set(int(scale) for scale in scales)))
    if not normalized_scales or normalized_scales[0] != 1 or any(scale < 1 for scale in normalized_scales):
        raise ValueError("scales must be positive and include 1")
    output_root = Path(output_root)
    root = Path(repo_root).resolve() if repo_root is not None else None

    def display(path: Path) -> str:
        resolved = Path(path).resolve()
        return resolved.relative_to(root).as_posix() if root is not None else resolved.name
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise ValueError(f"scaling output already exists and is not empty: {output_root}")
    base = load_corpus_snapshot(base_corpus)
    distractors = tuple(load_corpus_snapshot(path) for path in distractor_corpora)
    query_weights = _query_weights(inference_path)
    candidates = _candidate_rows(
        base=base,
        distractors=distractors,
        query_weights=query_weights,
        seed=seed,
    )
    order = _mixed_order(candidates, hard_fraction=hard_fraction)
    base_evidence = base.summary.evidence_count
    max_target = base_evidence * normalized_scales[-1]
    available_evidence = base_evidence + sum(int(row["evidence_count"]) for row in order)
    if available_evidence < max_target:
        raise ValueError(
            f"insufficient distractor evidence for {normalized_scales[-1]}x: "
            f"need {max_target}, have {available_evidence}"
        )

    selected: list[CorpusRecord] = []
    selected_evidence = 0
    selection_rows: list[dict[str, object]] = []
    scale_summaries: dict[str, object] = {}
    previous_ids: set[str] = set()
    order_index = 0
    for scale in normalized_scales:
        target = base_evidence * scale
        while base_evidence + selected_evidence < target:
            row = order[order_index]
            order_index += 1
            record = row["record"]
            if not isinstance(record, CorpusRecord):
                raise TypeError("internal scaling record is not a CorpusRecord")
            selected.append(record)
            selected_evidence += int(row["evidence_count"])
            selection_rows.append(
                {
                    "paragraph_id": row["paragraph_id"],
                    "source_dataset": record.source_dataset,
                    "source_split": record.source_split,
                    "hard_score": row["hard_score"],
                    "random_key": row["random_key"],
                    "selection_source": row["selection_source"],
                    "evidence_count": row["evidence_count"],
                    "first_included_scale": scale,
                }
            )
        snapshot = CorpusSnapshot.from_records([*base.records, *selected])
        current_ids = {record.paragraph_id for record in snapshot.records}
        if previous_ids and not previous_ids < current_ids:
            raise ValueError("scaling corpora are not strictly nested")
        previous_ids = current_ids
        scale_dir = output_root / f"scale_{scale}x"
        corpus_path = scale_dir / "corpus.jsonl"
        atomic_write_jsonl(
            corpus_path,
            (
                record.model_dump(mode="json")
                for record in sorted(snapshot.records, key=lambda item: item.paragraph_id)
            ),
        )
        validated = load_corpus_snapshot(corpus_path)
        summary = {
            "scale": scale,
            "target_evidence_count": target,
            "actual_evidence_count": validated.summary.evidence_count,
            "paragraph_count": validated.summary.paragraph_count,
            "corpus_id": validated.summary.corpus_id,
            "corpus_sha256": sha256_file(corpus_path),
            "base_paragraphs_preserved": len({row.paragraph_id for row in base.records} & current_ids),
            "all_base_paragraphs_preserved": {
                row.paragraph_id for row in base.records
            }.issubset(current_ids),
            "added_paragraph_count": len(selected),
            "source_paragraph_counts": dict(
                sorted(Counter(row.source_dataset for row in snapshot.records).items())
            ),
        }
        atomic_write_json(scale_dir / "manifest.json", summary)
        scale_summaries[str(scale)] = summary

    atomic_write_jsonl(output_root / "distractor_selection.jsonl", selection_rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "seed": seed,
        "hard_fraction": hard_fraction,
        "scales": list(normalized_scales),
        "base_corpus": {
            "path": display(base_corpus),
            "sha256": sha256_file(base_corpus),
            "corpus_id": base.summary.corpus_id,
            "paragraph_count": base.summary.paragraph_count,
            "evidence_count": base.summary.evidence_count,
        },
        "inference": {
            "path": display(inference_path),
            "sha256": sha256_file(inference_path),
            "visible_fields": ["dataset", "question"],
        },
        "distractor_corpora": [
            {
                "path": display(path),
                "sha256": sha256_file(path),
                "corpus_id": snapshot.summary.corpus_id,
            }
            for path, snapshot in zip(distractor_corpora, distractors, strict=True)
        ],
        "candidate_paragraph_count": len(candidates),
        "selected_paragraph_count": len(selection_rows),
        "scale_summaries": scale_summaries,
        "selection_sha256": sha256_file(output_root / "distractor_selection.jsonl"),
    }
    atomic_write_json(output_root / "scaling_manifest.json", manifest)
    return manifest


__all__ = ["build_scaling_corpora"]
