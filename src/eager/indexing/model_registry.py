from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Annotated, Iterable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from eager.config import canonical_json_bytes
from eager.indexing.errors import ModelRegistryError
from eager.runtime.io import DataFormatError, read_json, sha256_file


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Name = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ModelArtifactPin(BaseModel):
    """One required runtime file; unlisted snapshot files are allowed."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    path: NonEmptyText
    bytes: int = Field(ge=1)
    sha256: Sha256

    @model_validator(mode="after")
    def require_safe_relative_path(self) -> "ModelArtifactPin":
        path = PurePosixPath(self.path)
        if path.is_absolute() or "\\" in self.path or ":" in self.path:
            raise ValueError("model artifact path must be a relative POSIX path")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("model artifact path contains an unsafe segment")
        if path.as_posix() != self.path:
            raise ValueError("model artifact path must use canonical POSIX separators")
        return self


def artifact_manifest_sha256(artifacts: Iterable[ModelArtifactPin]) -> str:
    ordered = sorted(artifacts, key=lambda item: item.path)
    payload = [artifact.model_dump(mode="json") for artifact in ordered]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def model_pin_sha256(
    *,
    name: str,
    role: str,
    model_id: str,
    revision: str,
    artifact_manifest_sha256: str,
) -> str:
    payload = {
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "model_id": model_id,
        "name": name,
        "revision": revision,
        "role": role,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class ModelPin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    name: Name
    role: Name
    model_id: NonEmptyText
    revision: Revision
    artifacts: tuple[ModelArtifactPin, ...] = ()
    artifact_manifest_sha256: Sha256
    pin_sha256: Sha256

    @model_validator(mode="after")
    def verify_pin_hash(self) -> "ModelPin":
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError(f"duplicate artifact paths for {self.name}")
        folded_paths = [path.casefold() for path in paths]
        if len(folded_paths) != len(set(folded_paths)):
            raise ValueError(f"artifact paths conflict case-insensitively for {self.name}")
        if paths != sorted(paths):
            raise ValueError(f"artifact paths must be sorted for {self.name}")
        expected_manifest = artifact_manifest_sha256(self.artifacts)
        if self.artifact_manifest_sha256 != expected_manifest:
            raise ValueError(
                f"artifact_manifest_sha256 mismatch for {self.name}: "
                f"expected {expected_manifest}, got {self.artifact_manifest_sha256}"
            )
        expected = model_pin_sha256(
            name=self.name,
            role=self.role,
            model_id=self.model_id,
            revision=self.revision,
            artifact_manifest_sha256=self.artifact_manifest_sha256,
        )
        if self.pin_sha256 != expected:
            raise ValueError(
                f"pin_sha256 mismatch for {self.name}: expected {expected}, got {self.pin_sha256}"
            )
        return self


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    models: tuple[ModelPin, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> "ModelRegistry":
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model registry contains duplicate names")
        missing_manifests = [model.name for model in self.models if not model.artifacts]
        if missing_manifests:
            raise ValueError(
                "registered remote models require artifact manifests: "
                + ", ".join(missing_manifests)
            )
        return self

    def resolve(
        self,
        name: str,
        *,
        revision: str | None = None,
        pin_sha256: str | None = None,
    ) -> ModelPin:
        model = next((item for item in self.models if item.name == name), None)
        if model is None:
            raise ModelRegistryError(f"unknown registered model: {name}")
        if revision is not None and revision != model.revision:
            raise ModelRegistryError(
                f"model revision mismatch for {name}: expected {model.revision}, got {revision}"
            )
        if pin_sha256 is not None and pin_sha256 != model.pin_sha256:
            raise ModelRegistryError(f"model pin hash mismatch for {name}")
        return model


def load_model_registry(path: Path) -> ModelRegistry:
    try:
        return ModelRegistry.model_validate(read_json(path))
    except (DataFormatError, ValidationError) as exc:
        raise ModelRegistryError(f"invalid model registry {path}: {exc}") from exc


def verify_model_artifact(path: Path, *, expected_sha256: str) -> str:
    valid_hash = len(expected_sha256) == 64 and all(
        character in "0123456789abcdef" for character in expected_sha256
    )
    if not valid_hash:
        raise ModelRegistryError(
            "expected model artifact sha256 must be 64 lowercase hex characters"
        )
    try:
        if not path.is_file():
            raise OSError("path is not a file")
        actual = sha256_file(path)
    except OSError as exc:
        raise ModelRegistryError(f"cannot hash model artifact {path}: {exc}") from exc
    if actual != expected_sha256:
        raise ModelRegistryError(
            f"model artifact hash mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def verify_model_snapshot(
    snapshot_dir: Path,
    *,
    model: ModelPin,
    revision: str,
) -> str:
    if revision != model.revision:
        raise ModelRegistryError(
            f"model revision mismatch for {model.name}: "
            f"expected {model.revision}, got {revision}"
        )
    if not model.artifacts:
        raise ModelRegistryError(f"model {model.name} has no artifact manifest")
    snapshot_dir = Path(snapshot_dir)
    try:
        if not snapshot_dir.is_dir():
            raise OSError("path is not a directory")
    except OSError as exc:
        raise ModelRegistryError(f"cannot inspect model snapshot {snapshot_dir}: {exc}") from exc

    for artifact in model.artifacts:
        path = snapshot_dir.joinpath(*artifact.path.split("/"))
        try:
            if not path.is_file():
                raise OSError("required file is missing")
            actual_bytes = path.stat().st_size
        except OSError as exc:
            raise ModelRegistryError(
                f"cannot inspect model artifact {artifact.path}: {exc}"
            ) from exc
        if actual_bytes != artifact.bytes:
            raise ModelRegistryError(
                f"model artifact size mismatch for {artifact.path}: "
                f"expected {artifact.bytes}, got {actual_bytes}"
            )
        verify_model_artifact(path, expected_sha256=artifact.sha256)
    return model.artifact_manifest_sha256


__all__ = [
    "ModelArtifactPin",
    "ModelPin",
    "ModelRegistry",
    "ModelRegistryError",
    "artifact_manifest_sha256",
    "load_model_registry",
    "model_pin_sha256",
    "verify_model_artifact",
    "verify_model_snapshot",
]
