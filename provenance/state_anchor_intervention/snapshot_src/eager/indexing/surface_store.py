from __future__ import annotations

import hashlib
import inspect
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from eager.config import canonical_json_bytes
from eager.indexing.corpus import CorpusRecord, load_corpus_snapshot
from eager.indexing.errors import IndexValidationError, IndexingError
from eager.indexing.structured import (
    _mention_candidates,
    _normalize_alnum_surface,
    _surface_keys,
)
from eager.runtime.io import atomic_write_json, file_inventory_entry, read_json, sha256_file


_SCOPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_STORE_NAME = "surfaces.sqlite3"
_SUMMARY_NAME = "summary.json"
_STORE_FORMAT = "eager.sqlite-surface-v1"


def _ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode("utf-8")).hexdigest()


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA locking_mode=EXCLUSIVE;
        CREATE TABLE memberships (
            scope TEXT NOT NULL,
            paragraph_id TEXT NOT NULL,
            PRIMARY KEY (scope, paragraph_id)
        ) WITHOUT ROWID;
        CREATE INDEX memberships_by_paragraph ON memberships (paragraph_id, scope);
        CREATE TABLE paragraphs (
            paragraph_id TEXT PRIMARY KEY,
            title_tokens TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE title_token_postings (
            token TEXT NOT NULL,
            paragraph_id TEXT NOT NULL,
            PRIMARY KEY (token, paragraph_id)
        ) WITHOUT ROWID;
        CREATE TABLE surface_postings (
            surface_key TEXT NOT NULL,
            paragraph_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            alias TEXT NOT NULL,
            PRIMARY KEY (surface_key, paragraph_id, kind, alias)
        ) WITHOUT ROWID;
        CREATE INDEX surface_by_paragraph ON surface_postings (paragraph_id, surface_key);
        """
    )


def build_sqlite_surface_store(
    scopes: Mapping[str, Path],
    output_dir: Path,
    *,
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 128,
    n_process: int = 1,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if not scopes:
        raise IndexingError("surface store requires at least one corpus scope")
    if batch_size < 1:
        raise IndexingError("surface store batch_size must be positive")
    if n_process < 1:
        raise IndexingError("surface store n_process must be positive")
    for scope in scopes:
        if not _SCOPE_PATTERN.fullmatch(scope):
            raise IndexingError(f"invalid surface store scope: {scope!r}")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise IndexingError(f"surface store output already exists and is not empty: {output_dir}")
    try:
        import spacy
    except ImportError as exc:
        raise IndexingError("surface store construction requires spaCy") from exc
    try:
        nlp = spacy.load(spacy_model)
    except Exception as exc:
        raise IndexingError(f"cannot load spaCy model {spacy_model}: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / _STORE_NAME
    connection = sqlite3.connect(database_path)
    root = Path(repo_root).resolve() if repo_root is not None else None

    def display(path: Path) -> str:
        resolved = Path(path).resolve()
        return resolved.relative_to(root).as_posix() if root is not None else resolved.name

    records_by_id: dict[str, CorpusRecord] = {}
    scope_summaries: dict[str, Any] = {}
    try:
        _schema(connection)
        for scope, corpus_path in sorted(scopes.items()):
            snapshot = load_corpus_snapshot(Path(corpus_path))
            paragraph_ids = [record.paragraph_id for record in snapshot.records]
            connection.executemany(
                "INSERT INTO memberships(scope, paragraph_id) VALUES (?, ?)",
                ((scope, paragraph_id) for paragraph_id in paragraph_ids),
            )
            for record in snapshot.records:
                previous = records_by_id.setdefault(record.paragraph_id, record)
                if previous != record:
                    same_content = (
                        previous.title == record.title
                        and previous.text == record.text
                        and previous.sentences == record.sentences
                        and previous.source_dataset == record.source_dataset
                    )
                    if not same_content:
                        raise IndexingError(
                            f"conflicting scaling paragraph identity: {record.paragraph_id}"
                        )
            scope_summaries[scope] = {
                "corpus_path": display(Path(corpus_path)),
                "corpus_sha256": sha256_file(Path(corpus_path)),
                "corpus_id": snapshot.summary.corpus_id,
                "paragraph_count": snapshot.summary.paragraph_count,
                "evidence_count": snapshot.summary.evidence_count,
                "ordered_paragraph_id_sha256": _ordered_ids_sha256(paragraph_ids),
            }
        connection.commit()

        records = tuple(sorted(records_by_id.values(), key=lambda record: record.paragraph_id))
        for record in records:
            title_tokens = tuple(_normalize_alnum_surface(record.title).split())
            connection.execute(
                "INSERT INTO paragraphs(paragraph_id, title_tokens) VALUES (?, ?)",
                (record.paragraph_id, " ".join(title_tokens)),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO title_token_postings(token, paragraph_id) VALUES (?, ?)",
                ((token, record.paragraph_id) for token in title_tokens),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO surface_postings(surface_key, paragraph_id, kind, alias) "
                "VALUES (?, ?, 'title', ?)",
                ((key, record.paragraph_id, record.title) for key in _surface_keys(record.title)),
            )
        connection.commit()

        stream = (
            (sentence, (record.paragraph_id, record.title))
            for record in records
            for sentence in (record.sentences or (record.text,))
            if sentence
        )
        pending: list[tuple[str, str, str, str]] = []
        for doc, (paragraph_id, title) in nlp.pipe(
            stream,
            as_tuples=True,
            batch_size=batch_size,
            n_process=n_process,
        ):
            for candidate in _mention_candidates(doc, title=title):
                alias = str(candidate["text"])
                pending.extend(
                    (key, paragraph_id, "entity", alias)
                    for key in _surface_keys(alias)
                )
            if len(pending) >= 10000:
                connection.executemany(
                    "INSERT OR IGNORE INTO surface_postings"
                    "(surface_key, paragraph_id, kind, alias) VALUES (?, ?, ?, ?)",
                    pending,
                )
                connection.commit()
                pending.clear()
        if pending:
            connection.executemany(
                "INSERT OR IGNORE INTO surface_postings"
                "(surface_key, paragraph_id, kind, alias) VALUES (?, ?, ?, ?)",
                pending,
            )
            connection.commit()
        connection.execute("ANALYZE")
        connection.commit()
        counts = {
            "unique_paragraph_count": connection.execute(
                "SELECT COUNT(*) FROM paragraphs"
            ).fetchone()[0],
            "membership_count": connection.execute(
                "SELECT COUNT(*) FROM memberships"
            ).fetchone()[0],
            "surface_posting_count": connection.execute(
                "SELECT COUNT(*) FROM surface_postings"
            ).fetchone()[0],
            "surface_key_count": connection.execute(
                "SELECT COUNT(DISTINCT surface_key) FROM surface_postings"
            ).fetchone()[0],
            "title_token_posting_count": connection.execute(
                "SELECT COUNT(*) FROM title_token_postings"
            ).fetchone()[0],
        }
    except sqlite3.Error as exc:
        raise IndexingError(f"cannot build SQLite surface store: {exc}") from exc
    finally:
        connection.close()

    implementation_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": inspect.getsource(_schema),
                "builder": inspect.getsource(build_sqlite_surface_store),
                "mention_candidates": inspect.getsource(_mention_candidates),
                "surface_keys": inspect.getsource(_surface_keys),
            }
        )
    ).hexdigest()
    summary = {
        "schema_version": 1,
        "store_format": _STORE_FORMAT,
        "spacy_model": spacy_model,
        "spacy_version": spacy.__version__,
        "coreference_enabled": False,
        "batch_size": batch_size,
        "n_process": n_process,
        "implementation_sha256": implementation_sha256,
        "scopes": scope_summaries,
        **counts,
        "database": file_inventory_entry(database_path, root=output_dir),
    }
    atomic_write_json(output_dir / _SUMMARY_NAME, summary)
    return summary


def validate_sqlite_surface_store(store_dir: Path) -> dict[str, Any]:
    store_dir = Path(store_dir)
    summary = read_json(store_dir / _SUMMARY_NAME)
    if not isinstance(summary, dict) or summary.get("store_format") != _STORE_FORMAT:
        raise IndexValidationError(f"invalid SQLite surface store summary: {store_dir}")
    database = summary.get("database")
    if not isinstance(database, dict) or database.get("path") != _STORE_NAME:
        raise IndexValidationError("surface store database inventory is invalid")
    database_path = store_dir / _STORE_NAME
    if (
        database_path.stat().st_size != database.get("bytes")
        or sha256_file(database_path) != database.get("sha256")
    ):
        raise IndexValidationError("surface store database hash or size mismatch")
    try:
        connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise IndexValidationError(f"surface store integrity check failed: {integrity}")
        paragraph_count = connection.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        membership_count = connection.execute("SELECT COUNT(*) FROM memberships").fetchone()[0]
        posting_count = connection.execute("SELECT COUNT(*) FROM surface_postings").fetchone()[0]
    except sqlite3.Error as exc:
        raise IndexValidationError(f"cannot validate SQLite surface store: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    expected = (
        summary.get("unique_paragraph_count"),
        summary.get("membership_count"),
        summary.get("surface_posting_count"),
    )
    if (paragraph_count, membership_count, posting_count) != expected:
        raise IndexValidationError("surface store row counts differ from summary")
    return summary


def build_surface_scope_manifest(
    *,
    store_dir: Path,
    scope: str,
    corpus_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    summary = validate_sqlite_surface_store(store_dir)
    scope_summary = summary.get("scopes", {}).get(scope)
    if not isinstance(scope_summary, dict):
        raise IndexingError(f"unknown surface store scope: {scope}")
    corpus = load_corpus_snapshot(corpus_path)
    corpus_sha256 = sha256_file(corpus_path)
    paragraph_ids = [record.paragraph_id for record in corpus.records]
    if (
        scope_summary.get("corpus_sha256") != corpus_sha256
        or scope_summary.get("corpus_id") != corpus.summary.corpus_id
        or scope_summary.get("ordered_paragraph_id_sha256")
        != _ordered_ids_sha256(paragraph_ids)
    ):
        raise IndexingError("surface scope does not match the supplied corpus")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise IndexingError(f"surface scope output already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = Path(store_dir) / _STORE_NAME
    try:
        relative_database = database_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise IndexingError("surface store must be inside the repository") from exc
    manifest = {
        "schema_version": 1,
        "index_kind": "sqlite_surface_scope_v1",
        "scope": scope,
        "corpus_id": corpus.summary.corpus_id,
        "corpus_sha256": corpus_sha256,
        "paragraph_count": corpus.summary.paragraph_count,
        "evidence_count": corpus.summary.evidence_count,
        "ordered_paragraph_id_sha256": _ordered_ids_sha256(paragraph_ids),
        "store_path": relative_database,
        "store_sha256": summary["database"]["sha256"],
        "store_summary_sha256": sha256_file(Path(store_dir) / _SUMMARY_NAME),
    }
    atomic_write_json(output_dir / "store_manifest.json", manifest)
    atomic_write_json(
        output_dir / "index_summary.json",
        {
            "schema_version": 1,
            "index_kind": "sqlite_surface_scope_v1",
            "scope": scope,
            "paragraph_count": corpus.summary.paragraph_count,
            "evidence_count": corpus.summary.evidence_count,
            "store_sha256": summary["database"]["sha256"],
        },
    )
    return manifest


__all__ = [
    "build_sqlite_surface_store",
    "build_surface_scope_manifest",
    "validate_sqlite_surface_store",
]
