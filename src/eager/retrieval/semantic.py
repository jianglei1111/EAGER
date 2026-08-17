from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from eager.indexing import (
    CorpusSnapshot,
    DenseIndexSummary,
    SentenceEncoder,
    encode_tiny_text,
    load_sentence_encoder,
    load_tiny_index,
    validate_dense_index,
    verify_model_snapshot,
)


class SemanticIndex(Protocol):
    @property
    def evidence_ids(self) -> frozenset[str]: ...

    def score(self, query: str, evidence_id: str) -> float: ...

    def score_many(self, query: str, evidence_ids: tuple[str, ...]) -> dict[str, float]: ...

    def top_k(self, query: str, limit: int) -> tuple[tuple[str, float], ...]: ...


@dataclass(frozen=True, slots=True)
class TinySemanticIndex:
    dimensions: int
    vectors: dict[str, tuple[float, ...]]

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(self.vectors)

    @classmethod
    def load(
        cls,
        index_dir: Path,
        *,
        corpus: Path | CorpusSnapshot,
    ) -> "TinySemanticIndex":
        snapshot = load_tiny_index(index_dir, corpus=corpus)
        return cls(
            dimensions=snapshot.summary.dimensions,
            vectors={record.evidence_id: record.vector for record in snapshot.vectors},
        )

    def score(self, query: str, evidence_id: str) -> float:
        return self.score_many(query, (evidence_id,)).get(evidence_id, 0.0)

    def score_many(self, query: str, evidence_ids: tuple[str, ...]) -> dict[str, float]:
        query_vector = encode_tiny_text(query, dimensions=self.dimensions)
        scores: dict[str, float] = {}
        for evidence_id in evidence_ids:
            document = self.vectors.get(evidence_id)
            if document is None:
                scores[evidence_id] = 0.0
                continue
            cosine = sum(left * right for left, right in zip(query_vector, document))
            scores[evidence_id] = (
                max(0.0, float(cosine)) if math.isfinite(cosine) else 0.0
            )
        return scores

    def top_k(self, query: str, limit: int) -> tuple[tuple[str, float], ...]:
        if limit < 1:
            return ()
        scores = self.score_many(query, tuple(sorted(self.vectors)))
        return tuple(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        )


class DenseSemanticIndex:
    def __init__(
        self,
        *,
        summary: DenseIndexSummary,
        evidence_id_order: tuple[str, ...],
        embeddings: Any,
        ann_index: Any,
        encoder: SentenceEncoder,
        score_chunk_size: int = 8192,
    ) -> None:
        if score_chunk_size < 1:
            raise ValueError("score_chunk_size must be positive")
        self.summary = summary
        self.evidence_id_order = evidence_id_order
        self.embeddings = embeddings
        self.ann_index = ann_index
        self.encoder = encoder
        self.score_chunk_size = score_chunk_size
        self._positions = {
            evidence_id: position
            for position, evidence_id in enumerate(evidence_id_order)
        }
        self._query_cache: dict[str, Any] = {}
        self._encoder_lock = threading.RLock()
        self._ann_lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return self.summary.dimensions

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(self._positions)

    @classmethod
    def load(
        cls,
        index_dir: Path,
        *,
        corpus: Path | CorpusSnapshot,
        model_snapshot: Path,
        device: str = "cuda",
        encoder: SentenceEncoder | None = None,
    ) -> "DenseSemanticIndex":
        try:
            import numpy as np
        except ImportError as exc:
            raise ValueError("dense semantic retrieval requires numpy") from exc
        summary = validate_dense_index(index_dir, corpus=corpus)
        verify_model_snapshot(
            model_snapshot,
            model=summary.encoder,
            revision=summary.encoder.revision,
        )
        ids: list[str] = []
        with (Path(index_dir) / summary.evidence_ids.path).open(
            "r", encoding="utf-8-sig"
        ) as handle:
            for line in handle:
                ids.append(str(json.loads(line)["evidence_id"]))
        embeddings = np.load(
            Path(index_dir) / summary.vectors.path,
            mmap_mode="r",
            allow_pickle=False,
        )
        try:
            import faiss

            ann_index = faiss.read_index(str(Path(index_dir) / summary.ann.path))
            ann_index.hnsw.efSearch = summary.ann_config.ef_search
        except (ImportError, RuntimeError) as exc:
            raise ValueError("cannot load dense FAISS HNSW index") from exc
        runtime_encoder = encoder or load_sentence_encoder(model_snapshot, device=device)
        if runtime_encoder.get_sentence_embedding_dimension() != summary.dimensions:
            raise ValueError("dense query encoder dimension does not match the index")
        return cls(
            summary=summary,
            evidence_id_order=tuple(ids),
            embeddings=embeddings,
            ann_index=ann_index,
            encoder=runtime_encoder,
        )

    def _encode_query(self, query: str) -> Any:
        with self._encoder_lock:
            try:
                return self._query_cache[query]
            except KeyError:
                pass
            import numpy as np

            encoded = self.encoder.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            vector = np.asarray(encoded, dtype=np.float32)
            if vector.shape == (1, self.dimensions):
                vector = vector[0]
            if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
                raise ValueError("dense query encoder returned an invalid vector")
            norm = float(np.linalg.norm(vector))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2e-5):
                raise ValueError("dense query encoder returned a non-normalized vector")
            if len(self._query_cache) >= 32:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query] = vector
            return vector

    def score(self, query: str, evidence_id: str) -> float:
        return self.score_many(query, (evidence_id,)).get(evidence_id, 0.0)

    def score_many(self, query: str, evidence_ids: tuple[str, ...]) -> dict[str, float]:
        if not evidence_ids:
            return {}
        import numpy as np

        query_vector = self._encode_query(query)
        scores: dict[str, float] = {}
        for start in range(0, len(evidence_ids), self.score_chunk_size):
            chunk_ids = evidence_ids[start : start + self.score_chunk_size]
            known = [
                (evidence_id, self._positions[evidence_id])
                for evidence_id in chunk_ids
                if evidence_id in self._positions
            ]
            if known:
                positions = np.fromiter(
                    (position for _, position in known),
                    dtype=np.int64,
                    count=len(known),
                )
                values = np.asarray(self.embeddings[positions]) @ query_vector
                for (evidence_id, _), value in zip(known, values, strict=True):
                    score = float(value)
                    scores[evidence_id] = max(0.0, score) if math.isfinite(score) else 0.0
            for evidence_id in chunk_ids:
                scores.setdefault(evidence_id, 0.0)
        return scores

    def top_k(self, query: str, limit: int) -> tuple[tuple[str, float], ...]:
        if limit < 1:
            return ()
        import numpy as np

        query_vector = np.asarray(self._encode_query(query), dtype=np.float32).reshape(1, -1)
        with self._ann_lock:
            scores, positions = self.ann_index.search(
                query_vector,
                min(limit, len(self.evidence_id_order)),
            )
        output: list[tuple[str, float]] = []
        for position, score in zip(positions[0], scores[0], strict=True):
            if position < 0:
                continue
            value = float(score)
            output.append(
                (
                    self.evidence_id_order[int(position)],
                    max(0.0, value) if math.isfinite(value) else 0.0,
                )
            )
        return tuple(output)


__all__ = ["DenseSemanticIndex", "SemanticIndex", "TinySemanticIndex"]
