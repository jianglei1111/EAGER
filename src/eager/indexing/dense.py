from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from eager import __version__
from eager.config import canonical_json_bytes
from eager.indexing.corpus import CorpusSnapshot, CorpusSummary, load_corpus_snapshot
from eager.indexing.errors import IndexValidationError, IndexingError
from eager.indexing.model_registry import ModelPin, verify_model_snapshot
from eager.indexing.text_representation import load_evidence_texts
from eager.runtime.io import atomic_write_json, atomic_write_jsonl, sha256_file


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

_INDEX_FORMAT = "eager.dense-npy-v1"
_TEXT_VARIANT = "title_sep_sentence"
_RESOLVED_TEXT_VARIANT = "title_sep_resolved_sentence"
_DTYPE = "float32"
_NORMALIZED = True
_VALIDATION_CHUNK = 8192


class SentenceEncoder(Protocol):
    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


class DenseIndexArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    path: Literal["embeddings.npy"] = "embeddings.npy"
    bytes: int = Field(ge=1)
    sha256: Sha256
    record_count: int = Field(ge=1)


class DenseIdArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    path: Literal["evidence_ids.jsonl"] = "evidence_ids.jsonl"
    bytes: int = Field(ge=1)
    sha256: Sha256
    record_count: int = Field(ge=1)


class DenseTextArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    path: Literal["evidence_texts.jsonl"] = "evidence_texts.jsonl"
    bytes: int = Field(ge=1)
    sha256: Sha256
    source_sha256: Sha256
    record_count: int = Field(ge=1)


class DenseAnnArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    path: Literal["faiss_hnsw.index"] = "faiss_hnsw.index"
    bytes: int = Field(ge=1)
    sha256: Sha256
    record_count: int = Field(ge=1)


class DenseAnnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    format: Literal["faiss.IndexHNSWFlat.ip"] = "faiss.IndexHNSWFlat.ip"
    m: int = Field(ge=4, le=128)
    ef_construction: int = Field(ge=8, le=2048)
    ef_search: int = Field(ge=8, le=2048)


class DenseBuildProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    device: str = Field(min_length=1)
    batch_size: int = Field(ge=1)
    build_seconds: float = Field(ge=0.0)
    numpy_version: str = Field(min_length=1)
    sentence_transformers_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)


class DenseIndexSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: Literal[1] = 1
    index_format: Literal["eager.dense-npy-v1"] = _INDEX_FORMAT
    algorithm_sha256: Sha256
    package_version: str = Field(min_length=1)
    dimensions: int = Field(ge=8, le=4096)
    dtype: Literal["float32"] = _DTYPE
    normalized: Literal[True] = _NORMALIZED
    text_variant: Literal[
        "title_sep_sentence", "title_sep_resolved_sentence"
    ] = _TEXT_VARIANT
    parameters_sha256: Sha256
    corpus: CorpusSummary
    encoder: ModelPin
    vectors: DenseIndexArtifact
    evidence_ids: DenseIdArtifact
    representation: DenseTextArtifact | None = None
    ann: DenseAnnArtifact
    ann_config: DenseAnnConfig
    build: DenseBuildProvenance


def dense_semantic_text(title: str, sentence: str) -> str:
    return f"{title} [SEP] {sentence}"


_ALGORITHM_SPEC = {
    "format": _INDEX_FORMAT,
    "dtype": _DTYPE,
    "normalization": "l2",
    "text_variant": _TEXT_VARIANT,
    "ordering": "evidence_id",
    "similarity": "exact-cosine-via-inner-product",
    "ann": "faiss.IndexHNSWFlat-inner-product",
}
DENSE_ALGORITHM_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "spec": _ALGORITHM_SPEC,
            "semantic_text_source": inspect.getsource(dense_semantic_text),
        }
    )
).hexdigest()


def _parameters_sha256(
    model: ModelPin,
    dimensions: int,
    *,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    text_variant: str = _TEXT_VARIANT,
    representation_sha256: str | None = None,
) -> str:
    variant_parameters = (
        {}
        if text_variant == _TEXT_VARIANT and representation_sha256 is None
        else {
            "text_variant": text_variant,
            "representation_sha256": representation_sha256,
        }
    )
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "algorithm_sha256": DENSE_ALGORITHM_SHA256,
                "dimensions": dimensions,
                "model_pin_sha256": model.pin_sha256,
                "hnsw_m": hnsw_m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                **variant_parameters,
            }
        )
    ).hexdigest()


def default_huggingface_snapshot_path(model: ModelPin) -> Path:
    model_key = f"models--{model.model_id.replace('/', '--')}"
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / model_key
        / "snapshots"
        / model.revision
    )


def load_sentence_encoder(model_snapshot: Path, *, device: str) -> SentenceEncoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise IndexingError(
            "dense indexing requires the eager-rag[core] sentence-transformers dependency"
        ) from exc
    try:
        return SentenceTransformer(
            str(model_snapshot),
            device=device,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise IndexingError(f"cannot load dense encoder from {model_snapshot}: {exc}") from exc


def _coerce_snapshot(corpus: Path | CorpusSnapshot) -> CorpusSnapshot:
    return corpus if isinstance(corpus, CorpusSnapshot) else load_corpus_snapshot(Path(corpus))


def _environment_versions() -> tuple[str, str, str]:
    try:
        import numpy
        import sentence_transformers
        import torch
    except ImportError as exc:
        raise IndexingError("dense indexing dependencies are unavailable") from exc
    return numpy.__version__, sentence_transformers.__version__, torch.__version__


def build_dense_index(
    corpus: Path | CorpusSnapshot,
    output_dir: Path,
    *,
    model: ModelPin,
    model_snapshot: Path,
    device: str = "cuda",
    batch_size: int = 256,
    encoder: SentenceEncoder | None = None,
    show_progress_bar: bool = True,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
    evidence_texts: Mapping[str, str] | None = None,
    evidence_texts_source_sha256: str | None = None,
) -> DenseIndexSummary:
    if batch_size < 1:
        raise IndexingError("dense index batch_size must be positive")
    if model.role != "semantic_embedding":
        raise IndexingError("dense index model must have role=semantic_embedding")
    if not 4 <= hnsw_m <= 128:
        raise IndexingError("hnsw_m must be between 4 and 128")
    if not 8 <= ef_construction <= 2048 or not 8 <= ef_search <= 2048:
        raise IndexingError("HNSW ef values must be between 8 and 2048")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise IndexingError(f"dense index output already exists and is not empty: {output_dir}")
    verify_model_snapshot(model_snapshot, model=model, revision=model.revision)
    snapshot = _coerce_snapshot(corpus)
    units = tuple(sorted(snapshot.evidence_units, key=lambda item: item.evidence_id))
    unit_ids = {unit.evidence_id for unit in units}
    representation = dict(evidence_texts or {})
    if evidence_texts is not None and set(representation) != unit_ids:
        raise IndexingError("evidence text IDs do not match the canonical corpus")
    if any(not isinstance(text, str) or not text.strip() for text in representation.values()):
        raise IndexingError("evidence text representation contains an empty value")
    if evidence_texts is not None and evidence_texts_source_sha256 is None:
        raise IndexingError("alternate evidence texts require a source SHA256")
    text_variant = _RESOLVED_TEXT_VARIANT if evidence_texts is not None else _TEXT_VARIANT
    runtime_encoder = encoder or load_sentence_encoder(model_snapshot, device=device)
    dimensions = runtime_encoder.get_sentence_embedding_dimension()
    if dimensions is None or not 8 <= int(dimensions) <= 4096:
        raise IndexingError(f"invalid dense encoder dimension: {dimensions}")
    dimensions = int(dimensions)

    try:
        import numpy as np
    except ImportError as exc:
        raise IndexingError("dense indexing requires numpy") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "evidence_ids.jsonl"
    vectors_path = output_dir / "embeddings.npy"
    ann_path = output_dir / "faiss_hnsw.index"
    representation_path = output_dir / "evidence_texts.jsonl"
    atomic_write_jsonl(ids_path, ({"evidence_id": unit.evidence_id} for unit in units))
    if evidence_texts is not None:
        atomic_write_jsonl(
            representation_path,
            (
                {"evidence_id": unit.evidence_id, "text": representation[unit.evidence_id]}
                for unit in units
            ),
        )
    started = time.perf_counter()
    try:
        matrix = np.lib.format.open_memmap(
            vectors_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(units), dimensions),
        )
        for start in range(0, len(units), batch_size):
            batch = units[start : start + batch_size]
            encoded = runtime_encoder.encode(
                [
                    dense_semantic_text(
                        unit.title,
                        representation.get(unit.evidence_id, unit.text),
                    )
                    for unit in batch
                ],
                batch_size=batch_size,
                show_progress_bar=show_progress_bar and start == 0,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            values = np.asarray(encoded, dtype=np.float32)
            expected_shape = (len(batch), dimensions)
            if values.shape != expected_shape:
                raise IndexingError(
                    f"dense encoder returned shape {values.shape}, expected {expected_shape}"
                )
            if not np.isfinite(values).all():
                raise IndexingError("dense encoder returned non-finite values")
            norms = np.linalg.norm(values, axis=1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
                raise IndexingError("dense encoder returned non-normalized vectors")
            matrix[start : start + len(batch)] = values
        matrix.flush()
        del matrix
    except OSError as exc:
        raise IndexingError(f"cannot write dense index vectors: {exc}") from exc
    try:
        import faiss
    except ImportError as exc:
        raise IndexingError("dense indexing requires faiss-cpu") from exc
    try:
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        ann_index = faiss.IndexHNSWFlat(dimensions, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        ann_index.hnsw.efConstruction = ef_construction
        ann_index.hnsw.efSearch = ef_search
        for start in range(0, len(vectors), max(batch_size, 4096)):
            ann_index.add(np.asarray(vectors[start : start + max(batch_size, 4096)]))
        faiss.write_index(ann_index, str(ann_path))
    except (OSError, RuntimeError) as exc:
        raise IndexingError(f"cannot build FAISS HNSW index: {exc}") from exc
    build_seconds = time.perf_counter() - started
    numpy_version, sentence_transformers_version, torch_version = _environment_versions()
    summary = DenseIndexSummary(
        algorithm_sha256=DENSE_ALGORITHM_SHA256,
        package_version=__version__,
        dimensions=dimensions,
        parameters_sha256=_parameters_sha256(
            model,
            dimensions,
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            text_variant=text_variant,
            representation_sha256=(
                sha256_file(representation_path) if evidence_texts is not None else None
            ),
        ),
        text_variant=text_variant,
        corpus=snapshot.summary,
        encoder=model,
        vectors=DenseIndexArtifact(
            bytes=vectors_path.stat().st_size,
            sha256=sha256_file(vectors_path),
            record_count=len(units),
        ),
        evidence_ids=DenseIdArtifact(
            bytes=ids_path.stat().st_size,
            sha256=sha256_file(ids_path),
            record_count=len(units),
        ),
        representation=(
            DenseTextArtifact(
                bytes=representation_path.stat().st_size,
                sha256=sha256_file(representation_path),
                source_sha256=str(evidence_texts_source_sha256),
                record_count=len(units),
            )
            if evidence_texts is not None
            else None
        ),
        ann=DenseAnnArtifact(
            bytes=ann_path.stat().st_size,
            sha256=sha256_file(ann_path),
            record_count=len(units),
        ),
        ann_config=DenseAnnConfig(
            m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        ),
        build=DenseBuildProvenance(
            device=device,
            batch_size=batch_size,
            build_seconds=build_seconds,
            numpy_version=numpy_version,
            sentence_transformers_version=sentence_transformers_version,
            torch_version=torch_version,
        ),
    )
    atomic_write_json(output_dir / "summary.json", summary.model_dump(mode="json"))
    validate_dense_index(output_dir, corpus=snapshot, expected_model_pin=model)
    return summary


def _read_evidence_ids(path: Path) -> tuple[str, ...]:
    output: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IndexValidationError(
                        f"dense evidence IDs cannot contain blank lines: {path}:{line_number}"
                    )
                payload = json.loads(line)
                if not isinstance(payload, dict) or set(payload) != {"evidence_id"}:
                    raise IndexValidationError(
                        f"invalid dense evidence ID record: {path}:{line_number}"
                    )
                evidence_id = payload["evidence_id"]
                if not isinstance(evidence_id, str) or not evidence_id or any(
                    character.isspace() for character in evidence_id
                ):
                    raise IndexValidationError(
                        f"invalid dense evidence ID value: {path}:{line_number}"
                    )
                output.append(evidence_id)
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexValidationError(f"cannot read dense evidence IDs {path}: {exc}") from exc
    return tuple(output)


def validate_dense_index(
    index_dir: Path,
    *,
    corpus: Path | CorpusSnapshot | None = None,
    expected_model_pin: ModelPin | None = None,
    require_corpus: bool = False,
) -> DenseIndexSummary:
    try:
        import numpy as np
    except ImportError as exc:
        raise IndexValidationError("dense index validation requires numpy") from exc
    index_dir = Path(index_dir)
    summary_path = index_dir / "summary.json"
    try:
        summary = DenseIndexSummary.model_validate(
            json.loads(summary_path.read_text(encoding="utf-8-sig"))
        )
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise IndexValidationError(f"invalid dense index summary {summary_path}: {exc}") from exc
    if summary.algorithm_sha256 != DENSE_ALGORITHM_SHA256:
        raise IndexValidationError("dense index algorithm revision does not match this runtime")
    if summary.parameters_sha256 != _parameters_sha256(
        summary.encoder,
        summary.dimensions,
        hnsw_m=summary.ann_config.m,
        ef_construction=summary.ann_config.ef_construction,
        ef_search=summary.ann_config.ef_search,
        text_variant=summary.text_variant,
        representation_sha256=(
            summary.representation.sha256 if summary.representation is not None else None
        ),
    ):
        raise IndexValidationError("dense index parameter hash mismatch")
    if expected_model_pin is not None and summary.encoder != expected_model_pin:
        raise IndexValidationError("dense index encoder pin differs from the requested model pin")

    vectors_path = index_dir / summary.vectors.path
    ids_path = index_dir / summary.evidence_ids.path
    ann_path = index_dir / summary.ann.path
    for path, artifact, label in (
        (vectors_path, summary.vectors, "vectors"),
        (ids_path, summary.evidence_ids, "evidence IDs"),
        (ann_path, summary.ann, "ANN index"),
    ):
        try:
            actual_bytes = path.stat().st_size
            actual_sha256 = sha256_file(path)
        except OSError as exc:
            raise IndexValidationError(f"cannot inspect dense {label} {path}: {exc}") from exc
        if actual_bytes != artifact.bytes:
            raise IndexValidationError(f"dense {label} byte count mismatch")
        if actual_sha256 != artifact.sha256:
            raise IndexValidationError(f"dense {label} hash mismatch")

    evidence_ids = _read_evidence_ids(ids_path)
    if len(evidence_ids) != summary.evidence_ids.record_count:
        raise IndexValidationError("dense evidence ID record count mismatch")
    if evidence_ids != tuple(sorted(evidence_ids)) or len(evidence_ids) != len(set(evidence_ids)):
        raise IndexValidationError("dense evidence IDs must be unique and sorted")
    if summary.text_variant == _TEXT_VARIANT and summary.representation is not None:
        raise IndexValidationError("raw dense index cannot declare an alternate representation")
    if summary.text_variant == _RESOLVED_TEXT_VARIANT and summary.representation is None:
        raise IndexValidationError("resolved dense index is missing its text representation")
    if summary.representation is not None:
        representation_path = index_dir / summary.representation.path
        try:
            if representation_path.stat().st_size != summary.representation.bytes:
                raise IndexValidationError("dense evidence text byte count mismatch")
            if sha256_file(representation_path) != summary.representation.sha256:
                raise IndexValidationError("dense evidence text hash mismatch")
        except OSError as exc:
            raise IndexValidationError(
                f"cannot inspect dense evidence texts {representation_path}: {exc}"
            ) from exc
        loaded_texts = load_evidence_texts(
            representation_path,
            expected_ids=evidence_ids,
        )
        if len(loaded_texts) != summary.representation.record_count:
            raise IndexValidationError("dense evidence text record count mismatch")
    try:
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise IndexValidationError(f"cannot load dense vectors {vectors_path}: {exc}") from exc
    expected_shape = (summary.vectors.record_count, summary.dimensions)
    if vectors.shape != expected_shape or vectors.dtype != np.dtype(np.float32):
        raise IndexValidationError(
            f"dense vector shape/dtype mismatch: {vectors.shape}/{vectors.dtype}"
        )
    for start in range(0, len(vectors), _VALIDATION_CHUNK):
        chunk = np.asarray(vectors[start : start + _VALIDATION_CHUNK])
        if not np.isfinite(chunk).all():
            raise IndexValidationError("dense vectors contain non-finite values")
        norms = np.linalg.norm(chunk, axis=1)
        if any(not math.isclose(float(norm), 1.0, rel_tol=0.0, abs_tol=2e-5) for norm in norms):
            raise IndexValidationError("dense vectors are not normalized")
    try:
        import faiss

        ann_index = faiss.read_index(str(ann_path))
    except (ImportError, RuntimeError) as exc:
        raise IndexValidationError(f"cannot load dense ANN index {ann_path}: {exc}") from exc
    if (
        ann_index.d != summary.dimensions
        or ann_index.ntotal != summary.ann.record_count
        or ann_index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise IndexValidationError("dense ANN index structure does not match its summary")

    if require_corpus and corpus is None:
        raise IndexValidationError("full dense index validation requires the canonical corpus")
    if corpus is not None:
        snapshot = _coerce_snapshot(corpus)
        if snapshot.summary != summary.corpus:
            raise IndexValidationError("dense index corpus provenance does not match")
        expected_ids = tuple(
            sorted(unit.evidence_id for unit in snapshot.evidence_units)
        )
        if evidence_ids != expected_ids:
            raise IndexValidationError("dense index evidence IDs do not match the corpus")
    return summary


__all__ = [
    "DENSE_ALGORITHM_SHA256",
    "DenseBuildProvenance",
    "DenseAnnArtifact",
    "DenseAnnConfig",
    "DenseIdArtifact",
    "DenseIndexArtifact",
    "DenseIndexSummary",
    "DenseTextArtifact",
    "SentenceEncoder",
    "build_dense_index",
    "default_huggingface_snapshot_path",
    "dense_semantic_text",
    "load_sentence_encoder",
    "validate_dense_index",
]
