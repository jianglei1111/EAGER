from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from eager.data.base import (
    AdapterBatch,
    DatasetAdapterError,
    EvaluationBatch,
    InferenceBatch,
    require_mapping,
    require_text,
)
from eager.data.registry import (
    get_dataset_adapter,
    load_registered_split,
    resolve_registered_split,
    validate_split_overlaps,
)
from eager.indexing.corpus import (
    CorpusSnapshot,
    CorpusSummary,
    load_corpus_snapshot,
    validate_corpus,
)
from eager.runtime.io import (
    atomic_write_json,
    atomic_write_jsonl,
    file_inventory_entry,
    read_json,
    read_jsonl,
    sha256_file,
)
from eager.runtime.paths import PathContractError, repo_relative, resolve_path
from eager.schemas import EvaluationAnnotations, InferencePayload, QuestionHandle


PREPARE_MANIFEST_NAME = "prepare_manifest.json"
HANDLES_NAME = "handles.jsonl"
INFERENCE_NAME = "inference.jsonl"
ANNOTATIONS_NAME = "annotations.jsonl"
CORPUS_NAME = "corpus.jsonl"
CORPUS_SUMMARY_NAME = "corpus_summary.json"
_OUTPUT_NAMES = (
    HANDLES_NAME,
    INFERENCE_NAME,
    ANNOTATIONS_NAME,
    CORPUS_NAME,
    CORPUS_SUMMARY_NAME,
)


class DataWorkflowError(DatasetAdapterError):
    """Raised when prepared data violates the on-disk workflow contract."""


@dataclass(frozen=True, slots=True)
class PreparedDatasetSummary:
    dataset: str
    split: str
    record_count: int
    source_sha256: str
    prepared_dir: Path
    manifest_sha256: str
    anomalies: tuple[Mapping[str, Any], ...] = ()
    corpus_sha256: str | None = None
    paragraph_count: int = 0
    evidence_count: int = 0

    def public_dict(self, *, repo_root: Path) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "record_count": self.record_count,
            "source_sha256": self.source_sha256,
            "prepared_dir": repo_relative(self.prepared_dir, repo_root),
            "manifest_sha256": self.manifest_sha256,
            "anomaly_count": len(self.anomalies),
            "anomalies": [dict(anomaly) for anomaly in self.anomalies],
            "corpus_sha256": self.corpus_sha256,
            "paragraph_count": self.paragraph_count,
            "evidence_count": self.evidence_count,
        }


def _ids_sha256(values: Sequence[str]) -> str:
    payload = ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DataWorkflowError(f"{field} keys mismatch; missing={missing}, extra={extra}")


def _validate_annotation_support(
    annotations: Sequence[EvaluationAnnotations],
    corpus: CorpusSnapshot,
) -> None:
    paragraph_ids = {record.paragraph_id for record in corpus.records}
    sentence_ids = {unit.sentence_id for unit in corpus.evidence_units}
    titles = {record.title for record in corpus.records}
    for annotation in annotations:
        missing_paragraphs = set(annotation.supporting_paragraph_ids) - paragraph_ids
        missing_sentences = set(annotation.supporting_sentence_ids) - sentence_ids
        missing_titles = (
            set(annotation.supporting_titles) - titles
            if not annotation.supporting_paragraph_ids
            and not annotation.supporting_sentence_ids
            else set()
        )
        for step in annotation.official_decomposition:
            missing_paragraphs.update(set(step.supporting_paragraph_ids) - paragraph_ids)
            missing_sentences.update(set(step.supporting_sentence_ids) - sentence_ids)
        if missing_paragraphs or missing_sentences or missing_titles:
            raise DataWorkflowError(
                "evaluation support is absent from the retrieval corpus for "
                f"{annotation.handle_id}: paragraphs={sorted(missing_paragraphs)[:3]}, "
                f"sentences={sorted(missing_sentences)[:3]}, titles={sorted(missing_titles)[:3]}"
            )


def list_registered_splits(registry_path: Path, *, repo_root: Path) -> tuple[dict[str, Any], ...]:
    payload = require_mapping(read_json(registry_path), field="split registry")
    if payload.get("schema_version") not in {2, 3}:
        raise DataWorkflowError("split registry schema_version must be 2 or 3")
    validate_split_overlaps(registry_path, repo_root=repo_root)
    development = require_mapping(payload.get("development_splits", {}), field="development_splits")
    frozen = require_mapping(payload.get("frozen_profiles", {}), field="frozen_profiles")
    names = sorted((*development.keys(), *frozen.keys()))
    if len(names) != len(set(names)):
        raise DataWorkflowError("split registry reuses a name across development and frozen profiles")

    rows: list[dict[str, Any]] = []
    for name in names:
        split = resolve_registered_split(registry_path, str(name), repo_root=repo_root)
        rows.append(
            {
                "name": split.name,
                "dataset": split.dataset,
                "question_count": len(split.question_ids),
                "source_format": split.source_format,
                "source_path": repo_relative(split.source_path, repo_root),
                "source_sha256": split.expected_source_sha256,
                "registry_schema_version": split.registry_schema_version,
                "source_url": split.source_url,
                "source_revision": split.source_revision,
                "source_repository_revision": split.source_repository_revision,
                "source_artifact_verification": getattr(
                    split,
                    "source_artifact_verification",
                    None,
                ),
                "license_spdx": split.license_spdx,
                "license_reference": split.license_reference,
            }
        )
    return tuple(rows)


def _verify_source(split) -> None:
    try:
        actual_bytes = split.source_path.stat().st_size
    except OSError as exc:
        raise DataWorkflowError(f"cannot inspect dataset source {split.source_path}: {exc}") from exc
    if actual_bytes != split.expected_source_bytes:
        raise DataWorkflowError(
            f"dataset source size mismatch for {split.dataset}: "
            f"expected {split.expected_source_bytes}, got {actual_bytes}"
        )
    try:
        actual_sha256 = sha256_file(split.source_path)
    except OSError as exc:
        raise DataWorkflowError(f"cannot hash dataset source {split.source_path}: {exc}") from exc
    if actual_sha256 != split.expected_source_sha256:
        raise DataWorkflowError(f"dataset source hash mismatch for {split.dataset}")


def _safe_sha256(path: Path, *, field: str) -> str:
    try:
        return sha256_file(path)
    except OSError as exc:
        raise DataWorkflowError(f"cannot hash {field} {path}: {exc}") from exc


def acquire_registered_source(split, *, timeout_seconds: float = 120.0) -> None:
    """Download one missing, immutable source and publish it only after exact verification."""

    if split.source_path.exists():
        raise DataWorkflowError(f"dataset source already exists: {split.source_path}")
    if timeout_seconds <= 0:
        raise DataWorkflowError("download timeout must be positive")
    if split.registry_schema_version != 3 or split.source_url is None:
        raise DataWorkflowError("automatic source acquisition requires registry schema v3")
    verification = split.source_artifact_verification or ""
    if not verification.startswith("remote_"):
        raise DataWorkflowError(
            f"source is not approved for automatic acquisition: {verification or 'unverified'}"
        )
    lowered_url = split.source_url.lower()
    if re.search(r"/(?:main|master)(?:/|$)", lowered_url):
        raise DataWorkflowError("download URL must not use a floating main/master revision")
    if not split.source_revision or split.source_revision not in split.source_url:
        raise DataWorkflowError("download URL is not bound to the registered source revision")

    try:
        split.source_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{split.source_path.name}.",
            suffix=".download",
            dir=split.source_path.parent,
        )
    except OSError as exc:
        raise DataWorkflowError(
            f"cannot create download staging file for {split.dataset}: {exc}"
        ) from exc
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            try:
                with httpx.stream(
                    "GET",
                    split.source_url,
                    follow_redirects=True,
                    timeout=timeout_seconds,
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        byte_count += len(chunk)
                        if byte_count > split.expected_source_bytes:
                            raise DataWorkflowError(
                                f"downloaded source exceeds expected size for {split.dataset}"
                            )
                        digest.update(chunk)
                        output.write(chunk)
            except httpx.HTTPError as exc:
                raise DataWorkflowError(
                    f"cannot download registered source for {split.dataset}: {exc}"
                ) from exc
            output.flush()
            os.fsync(output.fileno())
        if byte_count != split.expected_source_bytes:
            raise DataWorkflowError(
                f"downloaded source size mismatch for {split.dataset}: "
                f"expected {split.expected_source_bytes}, got {byte_count}"
            )
        if digest.hexdigest() != split.expected_source_sha256:
            raise DataWorkflowError(f"downloaded source hash mismatch for {split.dataset}")
        os.replace(temp_path, split.source_path)
    except OSError as exc:
        raise DataWorkflowError(f"cannot publish dataset source {split.source_path}: {exc}") from exc
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError as exc:
            raise DataWorkflowError(f"cannot remove download staging file {temp_path}: {exc}") from exc
    _verify_source(split)


def _source_provenance(split) -> dict[str, Any]:
    return {
        "registry_schema_version": split.registry_schema_version,
        "source_url": split.source_url,
        "source_revision": split.source_revision,
        "source_repository_revision": split.source_repository_revision,
        "source_artifact_verification": getattr(
            split,
            "source_artifact_verification",
            None,
        ),
        "license_spdx": split.license_spdx,
        "license_reference": split.license_reference,
    }


def prepare_registered_dataset(
    registry_path: Path,
    split_name: str,
    output_dir: Path,
    *,
    repo_root: Path,
    anomalies_path: Path | None = None,
    download_missing: bool = False,
    download_timeout_seconds: float = 120.0,
) -> PreparedDatasetSummary:
    """Prepare a registered question split without mixing engine and evaluator records."""

    overlap_validation = validate_split_overlaps(registry_path, repo_root=repo_root)
    split = resolve_registered_split(registry_path, split_name, repo_root=repo_root)
    if anomalies_path is None and split.anomalies_path is not None:
        anomalies_path = split.anomalies_path
    if (
        split.anomalies_path is not None
        and anomalies_path is not None
        and anomalies_path.resolve() != split.anomalies_path.resolve()
    ):
        raise DataWorkflowError("anomaly manifest path differs from the split registry")
    if not split.source_path.exists() and download_missing:
        acquire_registered_source(split, timeout_seconds=download_timeout_seconds)
    if download_missing and download_timeout_seconds <= 0:
        raise DataWorkflowError("download timeout must be positive")
    _verify_source(split)
    if output_dir.exists():
        raise DataWorkflowError(f"prepared output already exists: {repo_relative(output_dir, repo_root)}")
    batch = load_registered_split(
        registry_path,
        split_name,
        repo_root=repo_root,
        verify_source=False,
    )

    handles = [handle.model_dump(mode="json") for handle in batch.inference.handles]
    inference = [payload.model_dump(mode="json") for payload in batch.inference.payloads]
    annotations = [
        batch.evaluation.by_handle_id[handle.handle_id].model_dump(mode="json")
        for handle in batch.inference.handles
    ]
    anomalies = _bound_anomalies(
        anomalies_path,
        dataset=split.dataset,
        source_sha256=split.expected_source_sha256,
        raw_dataset_ids={handle.raw_dataset_id for handle in batch.inference.handles},
    )
    rejected = [anomaly for anomaly in anomalies if anomaly["selected"]]
    if rejected:
        raise DataWorkflowError(
            f"registered split contains rejected dataset anomaly: {rejected[0]['anomaly_id']}"
        )
    adapter = get_dataset_adapter(split.dataset, split=split.name)
    corpus_snapshot = adapter.build_corpus_snapshot(
        adapter.read_rows(split.source_path),
        question_ids=split.question_ids,
    )
    _validate_annotation_support(
        tuple(batch.evaluation.by_handle_id[handle.handle_id] for handle in batch.inference.handles),
        corpus_snapshot,
    )
    anomaly_manifest = None
    if anomalies_path is not None:
        anomaly_manifest = {
            "path": repo_relative(anomalies_path, repo_root),
            "sha256": _safe_sha256(anomalies_path, field="dataset anomaly manifest"),
            "matched": list(anomalies),
        }

    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        ) as temp_name:
            staging_dir = Path(temp_name)
            atomic_write_jsonl(staging_dir / HANDLES_NAME, handles)
            atomic_write_jsonl(staging_dir / INFERENCE_NAME, inference)
            atomic_write_jsonl(staging_dir / ANNOTATIONS_NAME, annotations)
            atomic_write_jsonl(
                staging_dir / CORPUS_NAME,
                (
                    record.model_dump(mode="json")
                    for record in sorted(
                        corpus_snapshot.records,
                        key=lambda item: item.paragraph_id,
                    )
                ),
            )
            corpus_summary = validate_corpus(staging_dir / CORPUS_NAME)
            atomic_write_json(
                staging_dir / CORPUS_SUMMARY_NAME,
                corpus_summary.model_dump(mode="json"),
            )
            outputs = {
                name: file_inventory_entry(staging_dir / name, root=staging_dir)
                for name in _OUTPUT_NAMES
            }
            manifest = {
                "schema_version": 1,
                "dataset": split.dataset,
                "split": split.name,
                "record_count": len(handles),
                "ordered_question_id_sha256": _ids_sha256(split.question_ids),
                "overlap_validation": list(overlap_validation),
                "registry": {
                    "path": repo_relative(registry_path, repo_root),
                    "sha256": _safe_sha256(registry_path, field="split registry"),
                },
                "source": {
                    "path": repo_relative(split.source_path, repo_root),
                    "format": split.source_format,
                    "bytes": split.expected_source_bytes,
                    "sha256": split.expected_source_sha256,
                    "provenance": _source_provenance(split),
                },
                "anomaly_manifest": anomaly_manifest,
                "outputs": outputs,
                "isolation": {
                    "handles": HANDLES_NAME,
                    "engine_input": INFERENCE_NAME,
                    "evaluator_only": ANNOTATIONS_NAME,
                    "retrieval_corpus": CORPUS_NAME,
                    "corpus_summary": CORPUS_SUMMARY_NAME,
                },
            }
            atomic_write_json(staging_dir / PREPARE_MANIFEST_NAME, manifest)
            os.replace(staging_dir, output_dir)
    except OSError as exc:
        raise DataWorkflowError(f"cannot publish prepared dataset {output_dir}: {exc}") from exc

    return PreparedDatasetSummary(
        dataset=split.dataset,
        split=split.name,
        record_count=len(handles),
        source_sha256=split.expected_source_sha256,
        prepared_dir=output_dir,
        manifest_sha256=_safe_sha256(
            output_dir / PREPARE_MANIFEST_NAME,
            field="prepare manifest",
        ),
        anomalies=anomalies,
        corpus_sha256=corpus_summary.canonical_sha256,
        paragraph_count=corpus_summary.paragraph_count,
        evidence_count=corpus_summary.evidence_count,
    )


def _inventory_path(prepared_dir: Path, raw_entry: object, *, name: str) -> Path:
    entry = require_mapping(raw_entry, field=f"outputs.{name}")
    _exact_keys(entry, {"path", "bytes", "sha256"}, field=f"outputs.{name}")
    relative = require_text(entry.get("path"), field=f"outputs.{name}.path")
    if relative != name:
        raise DataWorkflowError(f"outputs.{name}.path must be {name!r}")
    try:
        path = resolve_path(relative, base=prepared_dir, must_exist=True)
    except (FileNotFoundError, PathContractError) as exc:
        raise DataWorkflowError(f"invalid prepared output path {relative}: {exc}") from exc
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise DataWorkflowError(f"cannot inspect prepared output {path}: {exc}") from exc
    try:
        expected_bytes = int(entry.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise DataWorkflowError(f"outputs.{name}.bytes must be an integer") from exc
    if actual_bytes != expected_bytes:
        raise DataWorkflowError(f"prepared output size mismatch: {name}")
    expected_sha256 = require_text(entry.get("sha256"), field=f"outputs.{name}.sha256")
    if _safe_sha256(path, field=f"prepared output {name}") != expected_sha256:
        raise DataWorkflowError(f"prepared output hash mismatch: {name}")
    return path


def _parse_models(records: Sequence[Any], model_type, *, field: str):
    parsed = []
    for index, record in enumerate(records):
        try:
            parsed.append(model_type.model_validate(record))
        except ValidationError as exc:
            raise DataWorkflowError(f"invalid {field}[{index}]: {exc}") from exc
    return tuple(parsed)


def load_prepared_batches(prepared_dir: Path) -> AdapterBatch:
    """Load the physically separated prepared records after provenance validation."""

    prepared_dir = Path(prepared_dir)
    handles = _parse_models(
        read_jsonl(prepared_dir / HANDLES_NAME),
        QuestionHandle,
        field="handles",
    )
    inference = _parse_models(
        read_jsonl(prepared_dir / INFERENCE_NAME),
        InferencePayload,
        field="inference",
    )
    annotations = _parse_models(
        read_jsonl(prepared_dir / ANNOTATIONS_NAME),
        EvaluationAnnotations,
        field="annotations",
    )
    if not handles:
        raise DataWorkflowError("prepared data must contain at least one question")
    if len(handles) != len(inference) or len(handles) != len(annotations):
        raise DataWorkflowError("prepared handles, inference, and annotations counts differ")
    for index, (handle, payload, annotation) in enumerate(
        zip(handles, inference, annotations, strict=True)
    ):
        if handle.dataset != payload.dataset:
            raise DataWorkflowError(
                f"prepared inference dataset differs from handles[{index}]"
            )
        if handle.handle_id != annotation.handle_id:
            raise DataWorkflowError(
                f"prepared annotation handle differs from handles[{index}]"
            )
    return AdapterBatch(
        inference=InferenceBatch(handles=handles, payloads=inference),
        evaluation=EvaluationBatch(
            by_handle_id={annotation.handle_id: annotation for annotation in annotations}
        ),
    )


def _manifest_repo_path(
    raw_path: object,
    *,
    field: str,
    repo_root: Path,
) -> Path:
    relative = require_text(raw_path, field=field)
    try:
        return resolve_path(relative, base=repo_root, must_exist=True)
    except (FileNotFoundError, PathContractError) as exc:
        raise DataWorkflowError(f"invalid {field}: {exc}") from exc


def _bound_anomalies(
    anomalies_path: Path | None,
    *,
    dataset: str,
    source_sha256: str,
    raw_dataset_ids: set[str],
) -> tuple[dict[str, Any], ...]:
    if anomalies_path is None:
        return ()
    payload = require_mapping(read_json(anomalies_path), field="dataset anomaly manifest")
    _exact_keys(payload, {"schema_version", "anomalies"}, field="dataset anomaly manifest")
    if payload.get("schema_version") != 1:
        raise DataWorkflowError("dataset anomaly manifest schema_version must be 1")
    raw_anomalies = payload.get("anomalies")
    if not isinstance(raw_anomalies, list):
        raise DataWorkflowError("dataset anomaly manifest anomalies must be an array")

    matched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    allowed = {
        "anomaly_id",
        "dataset",
        "source_sha256",
        "raw_dataset_id",
        "kind",
        "action",
        "note",
    }
    for index, raw_anomaly in enumerate(raw_anomalies):
        anomaly = require_mapping(raw_anomaly, field=f"anomalies[{index}]")
        extra = sorted(set(anomaly) - allowed)
        if extra:
            raise DataWorkflowError(f"anomalies[{index}] has unsupported keys: {extra}")
        anomaly_id = require_text(anomaly.get("anomaly_id"), field=f"anomalies[{index}].anomaly_id")
        if anomaly_id in seen_ids:
            raise DataWorkflowError(f"duplicate dataset anomaly ID: {anomaly_id}")
        seen_ids.add(anomaly_id)
        anomaly_dataset = require_text(anomaly.get("dataset"), field=f"anomalies[{index}].dataset")
        anomaly_sha256 = require_text(
            anomaly.get("source_sha256"), field=f"anomalies[{index}].source_sha256"
        ).lower()
        if re.fullmatch(r"[0-9a-f]{64}", anomaly_sha256) is None:
            raise DataWorkflowError(f"anomalies[{index}].source_sha256 must be a SHA256 digest")
        raw_dataset_id = anomaly.get("raw_dataset_id")
        if raw_dataset_id is not None:
            raw_dataset_id = require_text(
                raw_dataset_id, field=f"anomalies[{index}].raw_dataset_id"
            )
        kind = require_text(anomaly.get("kind"), field=f"anomalies[{index}].kind")
        action = require_text(
            anomaly.get("action", "reject"), field=f"anomalies[{index}].action"
        ).lower()
        if action != "reject":
            raise DataWorkflowError(
                f"anomalies[{index}].action must be 'reject'; gold repair is not supported"
            )
        note = anomaly.get("note")
        if note is not None:
            note = require_text(note, field=f"anomalies[{index}].note")

        if anomaly_dataset != dataset or anomaly_sha256 != source_sha256:
            continue
        normalized: dict[str, Any] = {
            "anomaly_id": anomaly_id,
            "dataset": anomaly_dataset,
            "source_sha256": anomaly_sha256,
            "kind": kind,
            "action": action,
            "selected": raw_dataset_id is None or raw_dataset_id in raw_dataset_ids,
        }
        if raw_dataset_id is not None:
            normalized["raw_dataset_id"] = raw_dataset_id
        if note is not None:
            normalized["note"] = note
        matched.append(normalized)
    return tuple(matched)


def validate_prepared_dataset(
    prepared_dir: Path,
    *,
    repo_root: Path,
    anomalies_path: Path | None = None,
) -> PreparedDatasetSummary:
    manifest_path = prepared_dir / PREPARE_MANIFEST_NAME
    manifest = require_mapping(read_json(manifest_path), field="prepare manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "dataset",
            "split",
            "record_count",
            "ordered_question_id_sha256",
            "overlap_validation",
            "registry",
            "source",
            "anomaly_manifest",
            "outputs",
            "isolation",
        },
        field="prepare manifest",
    )
    if manifest.get("schema_version") != 1:
        raise DataWorkflowError("prepare manifest schema_version must be 1")
    dataset = require_text(manifest.get("dataset"), field="dataset")
    split = require_text(manifest.get("split"), field="split")
    try:
        record_count = int(manifest.get("record_count", -1))
    except (TypeError, ValueError) as exc:
        raise DataWorkflowError("record_count must be an integer") from exc
    if record_count < 1:
        raise DataWorkflowError("record_count must be positive")

    registry = require_mapping(manifest.get("registry"), field="registry")
    _exact_keys(registry, {"path", "sha256"}, field="registry")
    registry_path = _manifest_repo_path(
        registry.get("path"), field="registry.path", repo_root=repo_root
    )
    registry_sha256 = require_text(registry.get("sha256"), field="registry.sha256")
    if _safe_sha256(registry_path, field="split registry") != registry_sha256:
        raise DataWorkflowError("split registry hash mismatch")
    overlap_validation = validate_split_overlaps(registry_path, repo_root=repo_root)
    recorded_overlap_validation = manifest.get("overlap_validation")
    if not isinstance(recorded_overlap_validation, list):
        raise DataWorkflowError("overlap_validation must be an array")
    if recorded_overlap_validation != list(overlap_validation):
        raise DataWorkflowError("prepare manifest overlap validation is stale or modified")
    registered = resolve_registered_split(registry_path, split, repo_root=repo_root)
    if registered.dataset != dataset:
        raise DataWorkflowError("prepare manifest dataset does not match the split registry")
    if record_count != len(registered.question_ids):
        raise DataWorkflowError("prepare manifest record_count does not match the split registry")
    ordered_id_sha256 = require_text(
        manifest.get("ordered_question_id_sha256"), field="ordered_question_id_sha256"
    )
    if ordered_id_sha256 != _ids_sha256(registered.question_ids):
        raise DataWorkflowError("prepare manifest ordered question ID hash mismatch")

    source = require_mapping(manifest.get("source"), field="source")
    _exact_keys(
        source,
        {"path", "format", "bytes", "sha256", "provenance"},
        field="source",
    )
    source_path = _manifest_repo_path(source.get("path"), field="source.path", repo_root=repo_root)
    try:
        source_bytes = int(source.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise DataWorkflowError("source.bytes must be an integer") from exc
    source_format = require_text(source.get("format"), field="source.format")
    source_sha256 = require_text(source.get("sha256"), field="source.sha256")
    if source_path != registered.source_path:
        raise DataWorkflowError("prepare manifest source path does not match the split registry")
    if source_format != registered.source_format:
        raise DataWorkflowError("prepare manifest source format does not match the split registry")
    if source_bytes != registered.expected_source_bytes:
        raise DataWorkflowError("prepare manifest source size does not match the split registry")
    if source_sha256 != registered.expected_source_sha256:
        raise DataWorkflowError("prepare manifest source hash does not match the split registry")
    provenance = require_mapping(source.get("provenance"), field="source.provenance")
    if dict(provenance) != _source_provenance(registered):
        raise DataWorkflowError("prepare manifest source provenance does not match the split registry")
    _verify_source(registered)

    isolation = require_mapping(manifest.get("isolation"), field="isolation")
    expected_isolation = {
        "handles": HANDLES_NAME,
        "engine_input": INFERENCE_NAME,
        "evaluator_only": ANNOTATIONS_NAME,
        "retrieval_corpus": CORPUS_NAME,
        "corpus_summary": CORPUS_SUMMARY_NAME,
    }
    if dict(isolation) != expected_isolation:
        raise DataWorkflowError("prepare manifest isolation mapping is invalid")

    outputs = require_mapping(manifest.get("outputs"), field="outputs")
    _exact_keys(outputs, set(_OUTPUT_NAMES), field="outputs")
    paths = {name: _inventory_path(prepared_dir, outputs[name], name=name) for name in _OUTPUT_NAMES}

    handles = _parse_models(read_jsonl(paths[HANDLES_NAME]), QuestionHandle, field="handles")
    inference = _parse_models(read_jsonl(paths[INFERENCE_NAME]), InferencePayload, field="inference")
    annotations = _parse_models(
        read_jsonl(paths[ANNOTATIONS_NAME]), EvaluationAnnotations, field="annotations"
    )
    corpus_snapshot = load_corpus_snapshot(paths[CORPUS_NAME])
    try:
        recorded_corpus_summary = CorpusSummary.model_validate(read_json(paths[CORPUS_SUMMARY_NAME]))
    except ValidationError as exc:
        raise DataWorkflowError(f"invalid corpus summary: {exc}") from exc
    if recorded_corpus_summary != corpus_snapshot.summary:
        raise DataWorkflowError("prepared corpus summary does not match corpus.jsonl")
    _validate_annotation_support(annotations, corpus_snapshot)
    if not (len(handles) == len(inference) == len(annotations) == record_count):
        raise DataWorkflowError("prepared record counts do not match the manifest")
    if any(handle.dataset != dataset or handle.split != split for handle in handles):
        raise DataWorkflowError("prepared handles do not match the manifest dataset/split")
    if any(payload.dataset != dataset for payload in inference):
        raise DataWorkflowError("prepared inference payloads do not match the manifest dataset")
    handle_ids = tuple(handle.handle_id for handle in handles)
    annotation_ids = tuple(annotation.handle_id for annotation in annotations)
    if handle_ids != annotation_ids:
        raise DataWorkflowError("prepared annotations are not aligned with orchestration handles")

    expected_batch = load_registered_split(
        registry_path,
        split,
        repo_root=repo_root,
        verify_source=False,
    )
    expected_handles = expected_batch.inference.handles
    expected_inference = expected_batch.inference.payloads
    expected_annotations = tuple(
        expected_batch.evaluation.by_handle_id[handle.handle_id] for handle in expected_handles
    )
    if handles != expected_handles:
        raise DataWorkflowError("prepared handles do not match the registered source split")
    if inference != expected_inference:
        raise DataWorkflowError("prepared inference does not match the registered source split")
    if annotations != expected_annotations:
        raise DataWorkflowError("prepared annotations do not match the registered source split")
    adapter = get_dataset_adapter(registered.dataset, split=registered.name)
    expected_corpus = adapter.build_corpus_snapshot(
        adapter.read_rows(registered.source_path),
        question_ids=registered.question_ids,
    )
    expected_corpus_records = tuple(
        sorted(expected_corpus.records, key=lambda item: item.paragraph_id)
    )
    if corpus_snapshot.records != expected_corpus_records:
        raise DataWorkflowError("prepared corpus does not match the registered source split")

    raw_anomaly_manifest = manifest.get("anomaly_manifest")
    if raw_anomaly_manifest is None:
        if registered.anomalies_path is not None:
            raise DataWorkflowError("prepare manifest is missing the registered anomaly manifest")
        if anomalies_path is not None:
            raise DataWorkflowError("prepared data is not bound to the supplied anomaly manifest")
        bound_anomalies: tuple[dict[str, Any], ...] = ()
    else:
        anomaly_manifest = require_mapping(raw_anomaly_manifest, field="anomaly_manifest")
        _exact_keys(
            anomaly_manifest,
            {"path", "sha256", "matched"},
            field="anomaly_manifest",
        )
        bound_path = _manifest_repo_path(
            anomaly_manifest.get("path"),
            field="anomaly_manifest.path",
            repo_root=repo_root,
        )
        if anomalies_path is not None and bound_path != anomalies_path.resolve():
            raise DataWorkflowError("supplied anomaly manifest does not match prepared provenance")
        if registered.anomalies_path is not None and bound_path != registered.anomalies_path.resolve():
            raise DataWorkflowError("prepared anomaly manifest differs from the split registry")
        anomaly_sha256 = require_text(
            anomaly_manifest.get("sha256"), field="anomaly_manifest.sha256"
        )
        if _safe_sha256(bound_path, field="dataset anomaly manifest") != anomaly_sha256:
            raise DataWorkflowError("dataset anomaly manifest hash mismatch")
        bound_anomalies = _bound_anomalies(
            bound_path,
            dataset=dataset,
            source_sha256=source_sha256,
            raw_dataset_ids={handle.raw_dataset_id for handle in handles},
        )
        recorded_anomalies = anomaly_manifest.get("matched")
        if not isinstance(recorded_anomalies, list):
            raise DataWorkflowError("anomaly_manifest.matched must be an array")
        if recorded_anomalies != list(bound_anomalies):
            raise DataWorkflowError("prepare manifest matched anomaly records are stale or modified")

    rejected = [anomaly for anomaly in bound_anomalies if anomaly["selected"]]
    if rejected:
        raise DataWorkflowError(
            f"prepared split contains rejected dataset anomaly: {rejected[0]['anomaly_id']}"
        )

    anomalies = _bound_anomalies(
        anomalies_path,
        dataset=dataset,
        source_sha256=source_sha256,
        raw_dataset_ids={handle.raw_dataset_id for handle in handles},
    ) if anomalies_path is not None else bound_anomalies
    return PreparedDatasetSummary(
        dataset=dataset,
        split=split,
        record_count=record_count,
        source_sha256=source_sha256,
        prepared_dir=prepared_dir,
        manifest_sha256=_safe_sha256(manifest_path, field="prepare manifest"),
        anomalies=anomalies,
        corpus_sha256=corpus_snapshot.summary.canonical_sha256,
        paragraph_count=corpus_snapshot.summary.paragraph_count,
        evidence_count=corpus_snapshot.summary.evidence_count,
    )


__all__ = [
    "ANNOTATIONS_NAME",
    "CORPUS_NAME",
    "CORPUS_SUMMARY_NAME",
    "DataWorkflowError",
    "HANDLES_NAME",
    "INFERENCE_NAME",
    "PREPARE_MANIFEST_NAME",
    "PreparedDatasetSummary",
    "acquire_registered_source",
    "list_registered_splits",
    "prepare_registered_dataset",
    "validate_prepared_dataset",
]
