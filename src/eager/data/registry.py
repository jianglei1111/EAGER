from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eager.data.base import (
    AdapterBatch,
    DatasetAdapter,
    DatasetAdapterError,
    require_integer,
    require_mapping,
    require_sequence,
    require_text,
)
from eager.data.hotpotqa import HotpotQAAdapter
from eager.data.musique import MusiqueAdapter
from eager.data.twowiki import TwoWikiAdapter
from eager.runtime.io import read_json, sha256_file


_SOURCE_CONTRACTS = {
    "hotpotqa": {
        "format": "json_array",
        "id_field": "_id",
        "canonical_id": "identity",
    },
    "2wiki": {
        "format": "parquet",
        "id_field": "_id",
        "canonical_id": "identity",
    },
    "musique": {
        "format": "jsonl",
        "id_field": "id",
        "canonical_id": "prefix:musique_",
    },
}


@dataclass(frozen=True, slots=True)
class RegisteredSplit:
    name: str
    dataset: str
    source_path: Path
    source_format: str
    question_ids: tuple[str, ...]
    expected_source_bytes: int
    expected_source_sha256: str
    registry_schema_version: int
    source_url: str | None = None
    source_revision: str | None = None
    source_artifact_verification: str | None = None
    source_repository_revision: str | None = None
    license_spdx: str | None = None
    license_reference: str | None = None
    anomalies_path: Path | None = None


def get_dataset_adapter(dataset: str, *, split: str) -> DatasetAdapter:
    normalized = dataset.strip().lower()
    adapter_types: dict[str, type[DatasetAdapter]] = {
        "hotpotqa": HotpotQAAdapter,
        "2wiki": TwoWikiAdapter,
        "musique": MusiqueAdapter,
    }
    try:
        adapter_type = adapter_types[normalized]
    except KeyError as exc:
        raise DatasetAdapterError(f"unsupported dataset: {dataset}") from exc
    return adapter_type(split=split)


def _read_ids(path: Path) -> tuple[str, ...]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DatasetAdapterError(f"cannot read split IDs {path}: {exc}") from exc
    if payload.startswith(b"\xef\xbb\xbf"):
        raise DatasetAdapterError(f"split IDs must not contain a UTF-8 BOM: {path}")
    if b"\r" in payload:
        raise DatasetAdapterError(f"split IDs must use LF line endings: {path}")
    if not payload.endswith(b"\n"):
        raise DatasetAdapterError(f"split IDs must end with a final LF: {path}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetAdapterError(f"split IDs must be valid UTF-8: {path}: {exc}") from exc
    lines = text[:-1].split("\n")
    if not lines or any(not line or line != line.strip() for line in lines):
        raise DatasetAdapterError(f"split IDs must contain one exact non-empty ID per line: {path}")
    if len(lines) != len(set(lines)):
        raise DatasetAdapterError(f"split IDs contain duplicates: {path}")
    return tuple(lines)


def _ids_sha256(question_ids: tuple[str, ...], *, ordered: bool) -> str:
    values = question_ids if ordered else tuple(sorted(question_ids))
    payload = ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_repo_path(repo_root: Path, raw_path: object, *, field: str) -> Path:
    text = require_text(raw_path, field=field)
    path = Path(text)
    if path.is_absolute():
        raise DatasetAdapterError(f"{field} must be repository-relative")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise DatasetAdapterError(f"{field} escapes the repository") from exc
    return resolved


def _require_sha256(value: object, *, field: str) -> str:
    digest = require_text(value, field=field).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise DatasetAdapterError(f"{field} must be a 64-character lowercase SHA256")
    return digest


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetAdapterError(f"{field} must be boolean")
    return value


def _validate_split_policy(
    split: Any,
    *,
    field: str,
    expected_purpose: str,
    tuning_allowed: bool,
    model_selection_allowed: bool,
) -> None:
    purpose = require_text(split.get("purpose"), field=f"{field}.purpose")
    if purpose != expected_purpose:
        raise DatasetAdapterError(f"{field}.purpose must be {expected_purpose!r}")
    if _require_bool(split.get("tuning_allowed"), field=f"{field}.tuning_allowed") is not tuning_allowed:
        raise DatasetAdapterError(f"{field}.tuning_allowed must be {tuning_allowed}")
    if _require_bool(
        split.get("model_selection_allowed"), field=f"{field}.model_selection_allowed"
    ) is not model_selection_allowed:
        raise DatasetAdapterError(
            f"{field}.model_selection_allowed must be {model_selection_allowed}"
        )


def _source_provenance(
    source: dict[str, Any] | Any,
    *,
    source_key: str,
    schema_version: int,
    repo_root: Path,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    Path | None,
]:
    if schema_version == 2:
        return None, None, None, None, None, None, None

    upstream = require_mapping(source.get("upstream"), field=f"raw_sources.{source_key}.upstream")
    source_url = require_text(
        upstream.get("download_url"), field=f"raw_sources.{source_key}.upstream.download_url"
    )
    if not source_url.startswith(("https://", "http://")):
        raise DatasetAdapterError(
            f"raw_sources.{source_key}.upstream.download_url must be an HTTP(S) URL"
        )
    source_revision = require_text(
        upstream.get("revision"), field=f"raw_sources.{source_key}.upstream.revision"
    )
    artifact_verification = require_text(
        upstream.get("artifact_verification"),
        field=f"raw_sources.{source_key}.upstream.artifact_verification",
    )
    repository_revision = require_text(
        upstream.get("repository_revision"),
        field=f"raw_sources.{source_key}.upstream.repository_revision",
    ).lower()
    if re.fullmatch(r"[0-9a-f]{40}", repository_revision) is None:
        raise DatasetAdapterError(
            f"raw_sources.{source_key}.upstream.repository_revision must be a 40-character commit"
        )

    license_payload = require_mapping(
        source.get("license"), field=f"raw_sources.{source_key}.license"
    )
    status = require_text(
        license_payload.get("status"), field=f"raw_sources.{source_key}.license.status"
    )
    if status != "verified":
        raise DatasetAdapterError(f"raw_sources.{source_key}.license.status must be 'verified'")
    license_spdx = require_text(
        license_payload.get("spdx"), field=f"raw_sources.{source_key}.license.spdx"
    )
    license_reference = require_text(
        license_payload.get("reference"), field=f"raw_sources.{source_key}.license.reference"
    )
    if not license_reference.startswith(("https://", "http://")):
        raise DatasetAdapterError(
            f"raw_sources.{source_key}.license.reference must be an HTTP(S) URL"
        )

    raw_anomalies = source.get("known_anomalies_path")
    anomalies_path = None
    if raw_anomalies is not None:
        anomalies_path = _resolve_repo_path(
            repo_root,
            raw_anomalies,
            field=f"raw_sources.{source_key}.known_anomalies_path",
        )
    return (
        source_url,
        source_revision,
        artifact_verification,
        repository_revision,
        license_spdx,
        license_reference,
        anomalies_path,
    )


def resolve_registered_split(
    registry_path: Path,
    split_name: str,
    *,
    repo_root: Path | None = None,
) -> RegisteredSplit:
    payload = require_mapping(read_json(registry_path), field="split registry")
    schema_version = require_integer(payload.get("schema_version"), field="split registry.schema_version")
    if schema_version not in {2, 3}:
        raise DatasetAdapterError("split registry schema_version must be 2 or 3")
    root = (repo_root or registry_path.resolve().parents[2]).resolve()
    development = require_mapping(payload.get("development_splits", {}), field="development_splits")
    frozen = require_mapping(payload.get("frozen_profiles", {}), field="frozen_profiles")
    raw_sources = require_mapping(payload.get("raw_sources"), field="raw_sources")

    if split_name in development:
        split = require_mapping(development[split_name], field=f"development_splits.{split_name}")
        if schema_version == 3:
            _validate_split_policy(
                split,
                field=f"development_splits.{split_name}",
                expected_purpose="development_only",
                tuning_allowed=True,
                model_selection_allowed=True,
            )
        dataset = require_text(split.get("dataset"), field=f"development_splits.{split_name}.dataset")
        source_key = require_text(
            split.get("source_key"), field=f"development_splits.{split_name}.source_key"
        )
        ids_path = _resolve_repo_path(
            root, split.get("ids_path"), field=f"development_splits.{split_name}.ids_path"
        )
        question_ids = _read_ids(ids_path)
        expected_count = require_integer(
            split.get("expected_count"), field=f"development_splits.{split_name}.expected_count"
        )
        ordered_hash = require_text(
            split.get("ordered_id_sha256"), field=f"development_splits.{split_name}.ordered_id_sha256"
        )
        set_hash = require_text(
            split.get("set_id_sha256"), field=f"development_splits.{split_name}.set_id_sha256"
        )
    elif split_name in frozen:
        split = require_mapping(frozen[split_name], field=f"frozen_profiles.{split_name}")
        if schema_version == 3:
            _validate_split_policy(
                split,
                field=f"frozen_profiles.{split_name}",
                expected_purpose="legacy_migration_only",
                tuning_allowed=False,
                model_selection_allowed=False,
            )
        dataset = require_text(split.get("dataset"), field=f"frozen_profiles.{split_name}.dataset")
        source_key = dataset
        source_entries = split.get("sources")
        if not isinstance(source_entries, list) or not source_entries:
            raise DatasetAdapterError(f"frozen profile {split_name} must contain sources")
        combined: list[str] = []
        for index, source_entry in enumerate(source_entries):
            source = require_mapping(source_entry, field=f"frozen_profiles.{split_name}.sources[{index}]")
            ids_path = _resolve_repo_path(
                root,
                source.get("ids_path"),
                field=f"frozen_profiles.{split_name}.sources[{index}].ids_path",
            )
            combined.extend(_read_ids(ids_path))
        question_ids = tuple(combined)
        if len(question_ids) != len(set(question_ids)):
            raise DatasetAdapterError(f"frozen profile {split_name} contains duplicate IDs across sources")
        expected_count = require_integer(
            split.get("expected_question_count"),
            field=f"frozen_profiles.{split_name}.expected_question_count",
        )
        ordered_hash = require_text(
            split.get("ordered_id_sha256"), field=f"frozen_profiles.{split_name}.ordered_id_sha256"
        )
        set_hash = require_text(
            split.get("set_id_sha256"), field=f"frozen_profiles.{split_name}.set_id_sha256"
        )
    else:
        raise DatasetAdapterError(f"unknown registered split: {split_name}")

    if len(question_ids) != expected_count:
        raise DatasetAdapterError(
            f"registered split {split_name} expected {expected_count} IDs, found {len(question_ids)}"
        )
    if _ids_sha256(question_ids, ordered=True) != ordered_hash:
        raise DatasetAdapterError(f"registered split ordered ID hash mismatch: {split_name}")
    if _ids_sha256(question_ids, ordered=False) != set_hash:
        raise DatasetAdapterError(f"registered split set ID hash mismatch: {split_name}")

    source = require_mapping(raw_sources.get(source_key), field=f"raw_sources.{source_key}")
    try:
        source_contract = _SOURCE_CONTRACTS[dataset]
    except KeyError as exc:
        raise DatasetAdapterError(f"unsupported dataset in registered split: {dataset}") from exc
    for field, expected_value in source_contract.items():
        actual_value = require_text(source.get(field), field=f"raw_sources.{source_key}.{field}")
        if actual_value != expected_value:
            raise DatasetAdapterError(
                f"raw_sources.{source_key}.{field} must be {expected_value!r}, got {actual_value!r}"
            )
    hash_mode = require_text(source.get("hash_mode"), field=f"raw_sources.{source_key}.hash_mode")
    if hash_mode != "byte_exact":
        raise DatasetAdapterError(f"raw_sources.{source_key}.hash_mode must be 'byte_exact'")
    expected_source_bytes = require_integer(
        source.get("expected_bytes"), field=f"raw_sources.{source_key}.expected_bytes"
    )
    if expected_source_bytes <= 0:
        raise DatasetAdapterError(f"raw_sources.{source_key}.expected_bytes must be positive")
    source_sha256 = _require_sha256(
        source.get("sha256"), field=f"raw_sources.{source_key}.sha256"
    )
    (
        source_url,
        source_revision,
        artifact_verification,
        repository_revision,
        license_spdx,
        license_reference,
        anomalies_path,
    ) = _source_provenance(
        source,
        source_key=source_key,
        schema_version=schema_version,
        repo_root=root,
    )
    return RegisteredSplit(
        name=split_name,
        dataset=dataset,
        source_path=_resolve_repo_path(root, source.get("path"), field=f"raw_sources.{source_key}.path"),
        source_format=source_contract["format"],
        question_ids=question_ids,
        expected_source_bytes=expected_source_bytes,
        expected_source_sha256=source_sha256,
        registry_schema_version=schema_version,
        source_url=source_url,
        source_revision=source_revision,
        source_artifact_verification=artifact_verification,
        source_repository_revision=repository_revision,
        license_spdx=license_spdx,
        license_reference=license_reference,
        anomalies_path=anomalies_path,
    )


def validate_split_overlaps(
    registry_path: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    payload = require_mapping(read_json(registry_path), field="split registry")
    schema_version = require_integer(payload.get("schema_version"), field="split registry.schema_version")
    if schema_version == 2:
        return ()
    if schema_version != 3:
        raise DatasetAdapterError("split registry schema_version must be 2 or 3")
    root = (repo_root or registry_path.resolve().parents[2]).resolve()
    constraints = require_sequence(
        payload.get("overlap_constraints"), field="overlap_constraints"
    )
    if not constraints:
        raise DatasetAdapterError("overlap_constraints must be non-empty")

    results: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, raw_constraint in enumerate(constraints):
        field = f"overlap_constraints[{index}]"
        constraint = require_mapping(raw_constraint, field=field)
        left_name = require_text(constraint.get("left"), field=f"{field}.left")
        right_name = require_text(constraint.get("right"), field=f"{field}.right")
        if left_name == right_name:
            raise DatasetAdapterError(f"{field} must compare two distinct splits")
        pair = tuple(sorted((left_name, right_name)))
        if pair in seen_pairs:
            raise DatasetAdapterError(f"duplicate overlap constraint: {pair[0]} / {pair[1]}")
        seen_pairs.add(pair)
        expected = require_integer(
            constraint.get("expected_intersection"), field=f"{field}.expected_intersection"
        )
        if expected < 0:
            raise DatasetAdapterError(f"{field}.expected_intersection must be non-negative")
        left = resolve_registered_split(registry_path, left_name, repo_root=root)
        right = resolve_registered_split(registry_path, right_name, repo_root=root)
        if left.dataset != right.dataset:
            raise DatasetAdapterError(f"{field} compares different datasets")
        actual = len(set(left.question_ids) & set(right.question_ids))
        if actual != expected:
            raise DatasetAdapterError(
                f"split overlap mismatch for {left_name} / {right_name}: expected {expected}, got {actual}"
            )
        results.append(
            {
                "left": left_name,
                "right": right_name,
                "expected_intersection": expected,
                "actual_intersection": actual,
            }
        )
    return tuple(results)


def load_registered_split(
    registry_path: Path,
    split_name: str,
    *,
    repo_root: Path | None = None,
    verify_source: bool = True,
) -> AdapterBatch:
    split = resolve_registered_split(registry_path, split_name, repo_root=repo_root)
    if verify_source:
        try:
            source_bytes = split.source_path.stat().st_size
        except OSError as exc:
            raise DatasetAdapterError(f"cannot inspect dataset source {split.source_path}: {exc}") from exc
        if source_bytes != split.expected_source_bytes:
            raise DatasetAdapterError(
                f"dataset source size mismatch for {split.dataset}: "
                f"expected {split.expected_source_bytes}, got {source_bytes}"
            )
        actual_hash = sha256_file(split.source_path)
        if actual_hash != split.expected_source_sha256:
            raise DatasetAdapterError(f"dataset source hash mismatch for {split.dataset}")
    adapter = get_dataset_adapter(split.dataset, split=split.name)
    return adapter.load(split.source_path, question_ids=split.question_ids)


__all__ = [
    "RegisteredSplit",
    "get_dataset_adapter",
    "load_registered_split",
    "resolve_registered_split",
    "validate_split_overlaps",
]
