from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import sqlite3
import threading
from typing import Any, Protocol

from eager.indexing import CorpusSnapshot
from eager.retrieval.anchors import generate_surface_keys, normalize_alnum_surface
from eager.schemas import QueryAnchor
from eager.runtime.io import read_json, sha256_file


class EntityMapper(Protocol):
    def map_anchor(self, anchor: QueryAnchor) -> "EntityMatch": ...

    def map_anchors(self, anchors: tuple[QueryAnchor, ...]) -> tuple["EntityMatch", ...]: ...


@dataclass(frozen=True, slots=True)
class EntityMatch:
    anchor_id: str
    paragraph_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EntityIndex:
    key_to_paragraph_ids: dict[str, tuple[str, ...]]
    paragraph_tokens: dict[str, frozenset[str]]
    token_to_paragraph_ids: dict[str, tuple[str, ...]]
    max_tokens_per_anchor: int = 8
    max_postings_per_token: int = 128
    max_matches_per_anchor: int = 8

    @classmethod
    def from_corpus(
        cls,
        corpus: CorpusSnapshot,
        *,
        max_tokens_per_anchor: int = 8,
        max_postings_per_token: int = 128,
        max_matches_per_anchor: int = 8,
    ) -> "EntityIndex":
        if min(max_tokens_per_anchor, max_postings_per_token, max_matches_per_anchor) < 1:
            raise ValueError("entity-index caps must be positive")
        keys: dict[str, set[str]] = {}
        paragraph_tokens: dict[str, frozenset[str]] = {}
        token_postings: dict[str, set[str]] = {}
        for record in corpus.records:
            for key in generate_surface_keys(record.title):
                keys.setdefault(key, set()).add(record.paragraph_id)
            title_tokens = frozenset(normalize_alnum_surface(record.title).split())
            paragraph_tokens[record.paragraph_id] = title_tokens
            for token in title_tokens:
                token_postings.setdefault(token, set()).add(record.paragraph_id)
        return cls(
            key_to_paragraph_ids={
                key: tuple(sorted(paragraph_ids)) for key, paragraph_ids in sorted(keys.items())
            },
            paragraph_tokens=paragraph_tokens,
            token_to_paragraph_ids={
                token: tuple(sorted(paragraph_ids)[:max_postings_per_token])
                for token, paragraph_ids in sorted(token_postings.items())
            },
            max_tokens_per_anchor=max_tokens_per_anchor,
            max_postings_per_token=max_postings_per_token,
            max_matches_per_anchor=max_matches_per_anchor,
        )

    @classmethod
    def from_surface_index(
        cls,
        index_dir: Path,
        *,
        corpus: CorpusSnapshot,
        max_tokens_per_anchor: int = 8,
        max_postings_per_token: int = 128,
        max_matches_per_anchor: int = 8,
    ) -> "EntityIndex":
        index_dir = Path(index_dir)
        title_payload: Any = read_json(index_dir / "title_surface_index.json")
        entity_payload: Any = read_json(index_dir / "entity_surface_index.json")
        if not isinstance(title_payload, dict) or not isinstance(entity_payload, dict):
            raise ValueError("structured surface indexes must be JSON objects")
        valid_paragraph_ids = {record.paragraph_id for record in corpus.records}
        keys: dict[str, set[str]] = defaultdict(set)
        for key, values in title_payload.items():
            if not isinstance(values, list):
                raise ValueError("title surface postings must be arrays")
            keys[str(key)].update(str(value) for value in values)
        for key, values in entity_payload.items():
            if not isinstance(values, list):
                raise ValueError("entity surface postings must be arrays")
            for value in values:
                if not isinstance(value, dict) or "paragraph_id" not in value:
                    raise ValueError("entity surface hit must contain paragraph_id")
                keys[str(key)].add(str(value["paragraph_id"]))
        unknown = set().union(*keys.values()) - valid_paragraph_ids if keys else set()
        if unknown:
            raise ValueError("structured surface index contains paragraphs outside the corpus")

        paragraph_tokens = {
            record.paragraph_id: frozenset(normalize_alnum_surface(record.title).split())
            for record in corpus.records
        }
        token_postings: dict[str, set[str]] = defaultdict(set)
        for key, paragraph_ids in keys.items():
            for token in normalize_alnum_surface(key).split():
                token_postings[token].update(paragraph_ids)
        return cls(
            key_to_paragraph_ids={
                key: tuple(sorted(paragraph_ids)[:max_postings_per_token])
                for key, paragraph_ids in sorted(keys.items())
            },
            paragraph_tokens=paragraph_tokens,
            token_to_paragraph_ids={
                token: tuple(sorted(paragraph_ids)[:max_postings_per_token])
                for token, paragraph_ids in sorted(token_postings.items())
            },
            max_tokens_per_anchor=max_tokens_per_anchor,
            max_postings_per_token=max_postings_per_token,
            max_matches_per_anchor=max_matches_per_anchor,
        )

    def map_anchor(self, anchor: QueryAnchor) -> EntityMatch:
        paragraph_ids: set[str] = set()
        anchor_keys = generate_surface_keys(anchor.surface)
        for key in anchor_keys:
            paragraph_ids.update(self.key_to_paragraph_ids.get(key, ()))
        if not paragraph_ids:
            anchor_tokens = frozenset(
                normalize_alnum_surface(anchor.surface).split()[: self.max_tokens_per_anchor]
            )
            if len(anchor_tokens) >= 2:
                postings = [
                    self.token_to_paragraph_ids[token]
                    for token in anchor_tokens
                    if token in self.token_to_paragraph_ids
                ]
                candidates = min(postings, key=lambda values: (len(values), values)) if postings else ()
                for paragraph_id in candidates:
                    title_tokens = self.paragraph_tokens[paragraph_id]
                    if anchor_tokens <= title_tokens or title_tokens <= anchor_tokens:
                        paragraph_ids.add(paragraph_id)
        return EntityMatch(
            anchor_id=anchor.anchor_id,
            paragraph_ids=tuple(sorted(paragraph_ids)[: self.max_matches_per_anchor]),
        )

    def map_anchors(self, anchors: tuple[QueryAnchor, ...]) -> tuple[EntityMatch, ...]:
        return tuple(match for anchor in anchors if (match := self.map_anchor(anchor)).paragraph_ids)


@dataclass(slots=True)
class SqliteEntityIndex:
    connection: sqlite3.Connection
    scope: str
    paragraph_tokens: dict[str, frozenset[str]]
    max_tokens_per_anchor: int = 8
    max_postings_per_token: int = 128
    max_matches_per_anchor: int = 8
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_store_manifest(
        cls,
        manifest_path: Path,
        *,
        corpus: CorpusSnapshot,
        repo_root: Path | None = None,
        max_tokens_per_anchor: int = 8,
        max_postings_per_token: int = 128,
        max_matches_per_anchor: int = 8,
    ) -> "SqliteEntityIndex":
        if min(max_tokens_per_anchor, max_postings_per_token, max_matches_per_anchor) < 1:
            raise ValueError("entity-index caps must be positive")
        manifest_path = Path(manifest_path)
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("index_kind") != "sqlite_surface_scope_v1":
            raise ValueError("invalid SQLite surface scope manifest")
        if repo_root is None:
            repo_root = next(
                (
                    parent
                    for parent in manifest_path.resolve().parents
                    if (parent / "pyproject.toml").is_file()
                ),
                None,
            )
        if repo_root is None:
            raise ValueError("cannot resolve repository root for SQLite surface store")
        store_path = (repo_root / str(manifest.get("store_path", ""))).resolve()
        try:
            store_path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError("SQLite surface store path escapes the repository") from exc
        if sha256_file(store_path) != manifest.get("store_sha256"):
            raise ValueError("SQLite surface store hash mismatch")
        paragraph_ids = [record.paragraph_id for record in corpus.records]
        ordered_hash = hashlib.sha256(
            ("\n".join(sorted(paragraph_ids)) + "\n").encode("utf-8")
        ).hexdigest()
        if (
            manifest.get("corpus_id") != corpus.summary.corpus_id
            or manifest.get("ordered_paragraph_id_sha256") != ordered_hash
        ):
            raise ValueError("SQLite surface scope does not match the corpus")
        connection = sqlite3.connect(str(store_path), check_same_thread=False)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
        scope = str(manifest["scope"])
        membership_count = connection.execute(
            "SELECT COUNT(*) FROM memberships WHERE scope = ?",
            (scope,),
        ).fetchone()[0]
        if membership_count != corpus.summary.paragraph_count:
            connection.close()
            raise ValueError("SQLite surface scope membership count mismatch")
        return cls(
            connection=connection,
            scope=scope,
            paragraph_tokens={
                record.paragraph_id: frozenset(normalize_alnum_surface(record.title).split())
                for record in corpus.records
            },
            max_tokens_per_anchor=max_tokens_per_anchor,
            max_postings_per_token=max_postings_per_token,
            max_matches_per_anchor=max_matches_per_anchor,
        )

    def _exact_postings(self, keys: tuple[str, ...]) -> tuple[str, ...]:
        if not keys:
            return ()
        placeholders = ",".join("?" for _ in keys)
        with self._lock:
            rows = self.connection.execute(
                "SELECT DISTINCT postings.paragraph_id "
                "FROM surface_postings AS postings "
                "JOIN memberships AS members "
                "ON members.paragraph_id = postings.paragraph_id AND members.scope = ? "
                f"WHERE postings.surface_key IN ({placeholders}) "
                "ORDER BY postings.paragraph_id LIMIT ?",
                (self.scope, *keys, self.max_matches_per_anchor),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _token_postings(self, token: str) -> tuple[str, ...]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT postings.paragraph_id "
                "FROM title_token_postings AS postings "
                "JOIN memberships AS members "
                "ON members.paragraph_id = postings.paragraph_id AND members.scope = ? "
                "WHERE postings.token = ? ORDER BY postings.paragraph_id LIMIT ?",
                (self.scope, token, self.max_postings_per_token),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def map_anchor(self, anchor: QueryAnchor) -> EntityMatch:
        anchor_keys = generate_surface_keys(anchor.surface)
        paragraph_ids = set(self._exact_postings(anchor_keys))
        if not paragraph_ids:
            anchor_tokens = frozenset(
                normalize_alnum_surface(anchor.surface).split()[: self.max_tokens_per_anchor]
            )
            if len(anchor_tokens) >= 2:
                postings = [self._token_postings(token) for token in anchor_tokens]
                candidates = min(postings, key=lambda values: (len(values), values)) if postings else ()
                for paragraph_id in candidates:
                    title_tokens = self.paragraph_tokens[paragraph_id]
                    if anchor_tokens <= title_tokens or title_tokens <= anchor_tokens:
                        paragraph_ids.add(paragraph_id)
        return EntityMatch(
            anchor_id=anchor.anchor_id,
            paragraph_ids=tuple(sorted(paragraph_ids)[: self.max_matches_per_anchor]),
        )

    def map_anchors(self, anchors: tuple[QueryAnchor, ...]) -> tuple[EntityMatch, ...]:
        return tuple(match for anchor in anchors if (match := self.map_anchor(anchor)).paragraph_ids)

    def close(self) -> None:
        with self._lock:
            self.connection.close()


__all__ = ["EntityIndex", "EntityMapper", "EntityMatch", "SqliteEntityIndex"]
