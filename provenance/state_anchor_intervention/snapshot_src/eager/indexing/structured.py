from __future__ import annotations

import hashlib
import inspect
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from eager.config import canonical_json_bytes
from eager.indexing.corpus import CorpusSnapshot, load_corpus_snapshot
from eager.indexing.errors import IndexingError
from eager.runtime.io import atomic_write_json, atomic_write_jsonl, file_inventory_entry, sha256_file


FORBIDDEN_STRUCTURED_FIELDS = frozenset(
    {
        "answer",
        "answer_aliases",
        "supporting_facts",
        "supporting_titles",
        "gold_supporting_question_ids",
        "gold_supporting_facts",
        "gold_type",
        "hop",
        "hop_count",
        "decomposition",
        "question_decomposition",
        "question_id",
        "question_ids",
        "raw_question_id",
        "relation",
    }
)


def _normalize_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("_", " ").strip().casefold()
    normalized = re.sub(r"\s*\(\s*", " (", normalized)
    normalized = re.sub(r"\s*\)\s*", ") ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return re.sub(r"^[^\w(]+|[^\w)]+$", "", normalized)


def _normalize_alnum_surface(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", _normalize_surface(text))).strip()


def _surface_keys(text: str) -> tuple[str, ...]:
    without_parenthetical = re.sub(r"\([^)]*\)", " ", text)
    keys = {
        _normalize_surface(text),
        _normalize_alnum_surface(text),
        _normalize_surface(without_parenthetical),
        _normalize_alnum_surface(without_parenthetical),
    }
    return tuple(sorted(key for key in keys if key))


def _mention_candidates(doc: Any, *, title: str) -> list[dict[str, Any]]:
    spans: dict[tuple[int, int], dict[str, Any]] = {}

    def add_span(span: Any, source: str) -> None:
        text = str(span.text).strip()
        normalized = _normalize_surface(text)
        if len(normalized) < 2:
            return
        key = (int(span.start_char), int(span.end_char))
        row = spans.setdefault(
            key,
            {
                "text": text,
                "normalized_text": normalized,
                "start_char": key[0],
                "end_char": key[1],
                "source_tags": [],
                "ner_labels": [],
            },
        )
        if source not in row["source_tags"]:
            row["source_tags"].append(source)
        label = str(getattr(span, "label_", "")).strip()
        if label and label not in row["ner_labels"]:
            row["ner_labels"].append(label)

    for entity in getattr(doc, "ents", ()):
        add_span(entity, "ner")

    try:
        for chunk in doc.noun_chunks:
            if len(chunk) <= 8 and any(token.pos_ in {"PROPN", "NOUN"} for token in chunk):
                add_span(chunk, "noun_chunk")
    except (AttributeError, ValueError):
        pass

    start: int | None = None
    for index, token in enumerate(doc):
        if token.pos_ == "PROPN":
            start = index if start is None else start
            continue
        if start is not None:
            add_span(doc[start:index], "proper_span")
            start = None
    if start is not None:
        add_span(doc[start : len(doc)], "proper_span")

    title_key = _normalize_surface(title)
    for row in spans.values():
        if row["normalized_text"] == title_key and "title" not in row["source_tags"]:
            row["source_tags"].append("title")
    return sorted(spans.values(), key=lambda row: (row["start_char"], row["end_char"], row["text"]))


def _cluster_key(surface: str) -> str:
    keys = _surface_keys(surface)
    if not keys:
        digest = hashlib.sha256(surface.encode("utf-8")).hexdigest()[:16]
        return f"unlexicalized_{digest}"
    alnum = [key for key in keys if key == _normalize_alnum_surface(key)]
    return min(alnum or list(keys), key=lambda key: (len(key), key))


def _ensure_safe_record(record: dict[str, Any]) -> None:
    overlap = FORBIDDEN_STRUCTURED_FIELDS & set(record)
    if overlap:
        raise IndexingError(f"structured retrieval record contains forbidden fields: {sorted(overlap)}")


def build_structured_resources(
    corpus: Path | CorpusSnapshot,
    output_dir: Path,
    *,
    surface_index_dir: Path | None = None,
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 128,
    window_radius: int = 1,
    max_surface_hits: int = 32,
    max_token_surfaces: int = 128,
    max_entity_sentences: int = 32,
    evidence_texts: Mapping[str, str] | None = None,
    evidence_texts_source_sha256: str | None = None,
) -> dict[str, Any]:
    if min(batch_size, max_surface_hits, max_token_surfaces, max_entity_sentences) < 1:
        raise IndexingError("structured preprocessing limits must be positive")
    if window_radius < 0:
        raise IndexingError("window_radius must be non-negative")
    snapshot = corpus if isinstance(corpus, CorpusSnapshot) else load_corpus_snapshot(Path(corpus))
    output_dir = Path(output_dir)
    surface_index_dir = Path(surface_index_dir or output_dir / "surface_index")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise IndexingError(f"structured output already exists and is not empty: {output_dir}")
    if surface_index_dir != output_dir / "surface_index" and surface_index_dir.exists() and any(surface_index_dir.iterdir()):
        raise IndexingError(f"surface index output already exists and is not empty: {surface_index_dir}")

    try:
        import spacy
    except ImportError as exc:
        raise IndexingError("structured preprocessing requires the eager-rag[core] spaCy dependency") from exc
    try:
        nlp = spacy.load(spacy_model)
    except Exception as exc:
        raise IndexingError(f"cannot load spaCy model {spacy_model}: {exc}") from exc

    paragraph_lookup = {record.paragraph_id: record for record in snapshot.records}
    sentence_rows: list[dict[str, Any]] = []
    mention_rows: list[dict[str, Any]] = []
    mentions_by_paragraph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sentence_rows_by_paragraph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_by_id = {unit.evidence_id: unit for unit in snapshot.evidence_units}
    representation = dict(evidence_texts or {})
    if evidence_texts is not None and set(representation) != set(evidence_by_id):
        raise IndexingError("evidence text IDs do not match the canonical corpus")
    if any(not isinstance(text, str) or not text.strip() for text in representation.values()):
        raise IndexingError("evidence text representation contains an empty value")
    if evidence_texts is not None and evidence_texts_source_sha256 is None:
        raise IndexingError("alternate evidence texts require a source SHA256")

    stream = (
        (representation.get(unit.evidence_id, unit.text), unit.evidence_id)
        for unit in snapshot.evidence_units
    )
    for doc, evidence_id in nlp.pipe(stream, as_tuples=True, batch_size=batch_size):
        unit = evidence_by_id[evidence_id]
        position = int(unit.sentence_id.rsplit("_s", 1)[1])
        candidates = _mention_candidates(doc, title=unit.title)
        local_mentions: list[dict[str, Any]] = []
        for candidate in candidates:
            mention_id = f"{unit.sentence_id}_m{len(local_mentions):03d}"
            row = {
                "mention_id": mention_id,
                "paragraph_id": unit.paragraph_id,
                "sentence_id": unit.sentence_id,
                "sentence_position": position,
                **candidate,
            }
            _ensure_safe_record(row)
            local_mentions.append(row)
            mentions_by_paragraph[unit.paragraph_id].append(row)
        sentence_row = {
            "sentence_id": unit.sentence_id,
            "paragraph_id": unit.paragraph_id,
            "title": unit.title,
            "position": position,
            "raw_sentence": unit.text,
            "resolved_sentence": representation.get(evidence_id, unit.text),
            "display_sentence": representation.get(evidence_id, unit.text),
            "mention_ids": [row["mention_id"] for row in local_mentions],
            "entity_cluster_ids": [],
        }
        _ensure_safe_record(sentence_row)
        sentence_rows.append(sentence_row)
        sentence_rows_by_paragraph[unit.paragraph_id].append(sentence_row)
        mention_rows.extend(local_mentions)

    sentence_lookup = {row["sentence_id"]: row for row in sentence_rows}
    entity_rows: list[dict[str, Any]] = []
    structured_paragraphs: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for paragraph_id in sorted(paragraph_lookup):
        record = paragraph_lookup[paragraph_id]
        paragraph_sentences = sorted(
            sentence_rows_by_paragraph[paragraph_id], key=lambda row: row["position"]
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for mention in mentions_by_paragraph[paragraph_id]:
            grouped[_cluster_key(mention["text"])].append(mention)
        title_key = _cluster_key(record.title)
        grouped.setdefault(title_key, [])

        paragraph_entity_ids: list[str] = []
        for cluster_position, key in enumerate(sorted(grouped)):
            mentions = grouped[key]
            aliases = sorted(
                {record.title if key == title_key else "", *(row["text"] for row in mentions)}
                - {""},
                key=lambda value: (-len(value), value.casefold(), value),
            )
            canonical_name = record.title if key == title_key else aliases[0]
            sentence_ids = sorted(
                {row["sentence_id"] for row in mentions}
                or ({paragraph_sentences[0]["sentence_id"]} if paragraph_sentences else set())
            )[:max_entity_sentences]
            entity_id = f"{paragraph_id}_e{cluster_position:04d}"
            entity_row = {
                "entity_cluster_id": entity_id,
                "paragraph_id": paragraph_id,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "mention_ids": [row["mention_id"] for row in mentions],
                "sentence_ids": sentence_ids,
            }
            _ensure_safe_record(entity_row)
            entity_rows.append(entity_row)
            paragraph_entity_ids.append(entity_id)
            for mention in mentions:
                mention["entity_cluster_id"] = entity_id
                sentence_lookup[mention["sentence_id"]]["entity_cluster_ids"].append(entity_id)

        sentence_ids = [row["sentence_id"] for row in paragraph_sentences]
        window_ids: list[str] = []
        for position, sentence in enumerate(paragraph_sentences):
            start = max(0, position - window_radius)
            end = min(len(paragraph_sentences), position + window_radius + 1)
            members = paragraph_sentences[start:end]
            window_id = f"{paragraph_id}_w{position:04d}"
            window_row = {
                "window_id": window_id,
                "paragraph_id": paragraph_id,
                "title": record.title,
                "center_sentence_id": sentence["sentence_id"],
                "sentence_ids": [row["sentence_id"] for row in members],
                "start_position": start,
                "end_position": end - 1,
                "raw_text": " ".join(row["raw_sentence"] for row in members),
                "resolved_text": " ".join(row["resolved_sentence"] for row in members),
                "display_text": " ".join(row["display_sentence"] for row in members),
            }
            _ensure_safe_record(window_row)
            window_rows.append(window_row)
            window_ids.append(window_id)

        paragraph_row = {
            "paragraph_id": paragraph_id,
            "title": record.title,
            "source_dataset": record.source_dataset,
            "source_split": record.source_split,
            "sentence_ids": sentence_ids,
            "entity_cluster_ids": paragraph_entity_ids,
            "window_ids": window_ids,
            "title_anchor": record.title,
        }
        _ensure_safe_record(paragraph_row)
        structured_paragraphs.append(paragraph_row)

    for row in sentence_rows:
        row["entity_cluster_ids"] = sorted(set(row["entity_cluster_ids"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "structured_paragraphs": output_dir / "structured_paragraphs.jsonl",
        "sentences": output_dir / "sentences.jsonl",
        "mentions": output_dir / "mentions.jsonl",
        "entity_clusters": output_dir / "entity_clusters.jsonl",
        "sentence_windows": output_dir / "sentence_windows.jsonl",
    }
    atomic_write_jsonl(artifacts["structured_paragraphs"], structured_paragraphs)
    atomic_write_jsonl(artifacts["sentences"], sentence_rows)
    atomic_write_jsonl(artifacts["mentions"], mention_rows)
    atomic_write_jsonl(artifacts["entity_clusters"], entity_rows)
    atomic_write_jsonl(artifacts["sentence_windows"], window_rows)

    surface_index_dir.mkdir(parents=True, exist_ok=True)
    surface_summary = _build_surface_index(
        structured_paragraphs=structured_paragraphs,
        sentences=sentence_rows,
        entities=entity_rows,
        windows=window_rows,
        output_dir=surface_index_dir,
        max_surface_hits=max_surface_hits,
        max_token_surfaces=max_token_surfaces,
    )
    implementation_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "mention_candidates": inspect.getsource(_mention_candidates),
                "cluster_key": inspect.getsource(_cluster_key),
                "builder": inspect.getsource(build_structured_resources),
                "surface_builder": inspect.getsource(_build_surface_index),
            }
        )
    ).hexdigest()
    summary = {
        "schema_version": 1,
        "corpus": snapshot.summary.model_dump(mode="json"),
        "spacy_model": spacy_model,
        "spacy_version": spacy.__version__,
        "coreference_enabled": evidence_texts is not None,
        "text_variant": (
            "resolved_sentence" if evidence_texts is not None else "raw_sentence"
        ),
        "evidence_texts_source_sha256": evidence_texts_source_sha256,
        "window_radius": window_radius,
        "paragraph_count": len(structured_paragraphs),
        "sentence_count": len(sentence_rows),
        "mention_count": len(mention_rows),
        "entity_cluster_count": len(entity_rows),
        "window_count": len(window_rows),
        "forbidden_fields": sorted(FORBIDDEN_STRUCTURED_FIELDS),
        "implementation_sha256": implementation_sha256,
        "surface_index": surface_summary,
        "artifacts": {
            name: file_inventory_entry(path, root=output_dir)
            for name, path in sorted(artifacts.items())
        },
    }
    atomic_write_json(output_dir / "structured_summary.json", summary)
    return summary


def _build_surface_index(
    *,
    structured_paragraphs: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    output_dir: Path,
    max_surface_hits: int,
    max_token_surfaces: int,
) -> dict[str, Any]:
    title_index: dict[str, set[str]] = defaultdict(set)
    entity_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    token_surfaces: dict[str, set[str]] = defaultdict(set)
    entity_to_sentence: dict[str, list[str]] = {}
    paragraph_to_window: dict[str, list[str]] = defaultdict(list)
    sentence_to_window: dict[str, list[str]] = defaultdict(list)

    for paragraph in structured_paragraphs:
        for key in _surface_keys(paragraph["title"]):
            title_index[key].add(paragraph["paragraph_id"])
            for token in _normalize_alnum_surface(key).split():
                token_surfaces[token].add(key)
    for entity in entities:
        entity_to_sentence[entity["entity_cluster_id"]] = entity["sentence_ids"]
        for alias in {entity["canonical_name"], *entity["aliases"]}:
            for key in _surface_keys(alias):
                entity_index[key].append(
                    {
                        "entity_cluster_id": entity["entity_cluster_id"],
                        "paragraph_id": entity["paragraph_id"],
                        "canonical_name": entity["canonical_name"],
                        "alias": alias,
                    }
                )
                for token in _normalize_alnum_surface(key).split():
                    token_surfaces[token].add(key)
    for window in windows:
        paragraph_to_window[window["paragraph_id"]].append(window["window_id"])
        for sentence_id in window["sentence_ids"]:
            sentence_to_window[sentence_id].append(window["window_id"])

    title_payload = {
        key: sorted(values)[:max_surface_hits] for key, values in sorted(title_index.items())
    }
    entity_payload = {
        key: sorted(
            values,
            key=lambda row: (
                row["paragraph_id"], row["entity_cluster_id"], row["canonical_name"], row["alias"]
            ),
        )[:max_surface_hits]
        for key, values in sorted(entity_index.items())
    }
    token_payload = {
        token: sorted(values)[:max_token_surfaces]
        for token, values in sorted(token_surfaces.items())
    }
    files: dict[str, Any] = {
        "title_surface_index.json": title_payload,
        "entity_surface_index.json": entity_payload,
        "token_surface_index.json": token_payload,
        "entity_to_sentence_map.json": dict(sorted(entity_to_sentence.items())),
        "paragraph_to_window_map.json": {
            key: sorted(set(values)) for key, values in sorted(paragraph_to_window.items())
        },
        "sentence_to_window_map.json": {
            key: sorted(set(values)) for key, values in sorted(sentence_to_window.items())
        },
    }
    for name, payload in files.items():
        atomic_write_json(output_dir / name, payload)
    atomic_write_jsonl(output_dir / "window_store.jsonl", windows)
    summary = {
        "schema_version": 1,
        "max_surface_hits": max_surface_hits,
        "max_token_surfaces": max_token_surfaces,
        "title_key_count": len(title_payload),
        "entity_key_count": len(entity_payload),
        "token_key_count": len(token_payload),
        "entity_count": len(entities),
        "sentence_count": len(sentences),
        "window_count": len(windows),
    }
    atomic_write_json(output_dir / "index_summary.json", summary)
    summary["index_summary_sha256"] = sha256_file(output_dir / "index_summary.json")
    return summary


__all__ = ["FORBIDDEN_STRUCTURED_FIELDS", "build_structured_resources"]
