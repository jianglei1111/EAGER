from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from eager.config import canonical_json_bytes


class ProtocolError(ValueError):
    """Raised when an experiment protocol is invalid or cannot be migrated."""


def _relative_repo_path(value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError("protocol paths must be repository-relative")
    normalized = candidate.as_posix()
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("protocol path cannot escape the repository")
    return normalized


class DatasetProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_config: str
    processed_sentences: str
    index_dir: str
    formal_profile: str = Field(min_length=1)
    development_split: str = Field(min_length=1)
    primary_support_unit: Literal["sentence", "paragraph"]
    secondary_support_unit: Literal["sentence", "paragraph"] | None = None

    @field_validator("question_config", "processed_sentences", "index_dir")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _relative_repo_path(value)


class DevelopmentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_queries: int = Field(ge=1)
    minimum_queries: int = Field(ge=1)
    maximum_queries: int = Field(ge=1)
    disjoint_from_formal: bool = True
    selection_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "DevelopmentPolicy":
        if self.minimum_queries > self.default_queries or self.default_queries > self.maximum_queries:
            raise ValueError("development query counts must satisfy minimum <= default <= maximum")
        return self


class EvaluationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frozen_test_queries: int = Field(ge=1)
    development: DevelopmentPolicy
    timing_queries: int = Field(ge=1)
    candidate_budgets: list[int]
    final_evidence_budgets: list[int]
    corpus_scales: list[int]
    random_seed: int = Field(ge=0)

    @field_validator("candidate_budgets", "final_evidence_budgets", "corpus_scales")
    @classmethod
    def positive_unique_sorted(cls, values: list[int]) -> list[int]:
        if not values or any(value <= 0 for value in values):
            raise ValueError("budget and scale lists must contain positive integers")
        if values != sorted(set(values)):
            raise ValueError("budget and scale lists must be sorted and unique")
        return values


class ExecutionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    purpose: str = Field(min_length=1)
    oracle_annotations_allowed: bool
    publishable_main_result: bool
    forbidden_controller_fields: list[str] = Field(default_factory=list)

    @field_validator("forbidden_controller_fields")
    @classmethod
    def unique_fields(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("forbidden controller fields must be unique")
        return values


class ProtocolProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    migrated_from_schema_version: int | None = None
    split_registry: str

    @field_validator("split_registry")
    @classmethod
    def validate_split_path(cls, value: str) -> str:
        return _relative_repo_path(value)


class ExperimentProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    objective: str = Field(min_length=1)
    datasets: dict[str, DatasetProtocol]
    evaluation: EvaluationProtocol
    candidate_variants: list[str]
    refinement_variants: list[str]
    policy_variants: list[str]
    required_metrics: list[str]
    execution_profiles: dict[str, ExecutionProfile]
    provenance: ProtocolProvenance

    @field_validator(
        "candidate_variants",
        "refinement_variants",
        "policy_variants",
        "required_metrics",
    )
    @classmethod
    def nonempty_unique(cls, values: list[str]) -> list[str]:
        if not values or any(not value.strip() for value in values):
            raise ValueError("protocol lists must be non-empty")
        if len(values) != len(set(values)):
            raise ValueError("protocol list values must be unique")
        return values

    @model_validator(mode="after")
    def require_profiles(self) -> "ExperimentProtocol":
        required = {"legacy_reproduction", "wsdm_strict"}
        missing = required - set(self.execution_profiles)
        if missing:
            raise ValueError(f"missing execution profiles: {sorted(missing)}")
        strict = self.execution_profiles["wsdm_strict"]
        if strict.oracle_annotations_allowed or not strict.publishable_main_result:
            raise ValueError("wsdm_strict must forbid oracle annotations and be publishable")
        return self


def _parse_development_range(value: Any) -> tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    match = re.search(r"(\d+)\s*[-–]\s*(\d+)", str(value))
    if match is None:
        raise ProtocolError(f"cannot migrate development_queries: {value!r}")
    minimum = int(match.group(1))
    maximum = int(match.group(2))
    return minimum, minimum, maximum


def migrate_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    version = int(payload.get("schema_version", 0))
    if version == 2:
        return dict(payload)
    if version != 1:
        raise ProtocolError(f"unsupported experiment protocol schema_version={version}")

    evaluation = dict(payload.get("evaluation", {}))
    default_queries, minimum_queries, maximum_queries = _parse_development_range(
        evaluation.pop("development_queries", 200)
    )
    random_seed = int(evaluation.get("random_seed", 0))
    evaluation["development"] = {
        "default_queries": default_queries,
        "minimum_queries": minimum_queries,
        "maximum_queries": maximum_queries,
        "disjoint_from_formal": True,
        "selection_seed": random_seed,
    }

    datasets: dict[str, Any] = {}
    formal_names = {
        "hotpotqa": "hotpotqa_main_1000",
        "2wiki": "2wiki_main_1000",
        "musique": "musique_main_1000",
    }
    development_names = {
        "hotpotqa": "hotpotqa_dev200",
        "2wiki": "2wiki_dev200",
        "musique": "musique_dev200",
    }
    for name, raw_dataset in dict(payload.get("datasets", {})).items():
        datasets[name] = {
            **dict(raw_dataset),
            "formal_profile": formal_names.get(name, f"{name}_formal"),
            "development_split": development_names.get(name, f"{name}_development"),
        }

    forbidden = [
        "answer",
        "supporting_facts",
        "supporting_titles",
        "type",
        "hop",
        "hop_count",
        "decomposition",
        "question_decomposition",
        "relation",
        "raw_question_id",
    ]
    return {
        **{key: value for key, value in payload.items() if key not in {"schema_version", "datasets", "evaluation"}},
        "schema_version": 2,
        "datasets": datasets,
        "evaluation": evaluation,
        "execution_profiles": {
            "legacy_reproduction": {
                "purpose": "Read-only mechanical comparison with historical oracle-assisted artifacts.",
                "oracle_annotations_allowed": True,
                "publishable_main_result": False,
                "forbidden_controller_fields": [],
            },
            "wsdm_strict": {
                "purpose": "Publishable unified online system with annotation-isolated inference.",
                "oracle_annotations_allowed": False,
                "publishable_main_result": True,
                "forbidden_controller_fields": forbidden,
            },
        },
        "provenance": {
            "migrated_from_schema_version": 1,
            "split_registry": "wsdm/config/split_registry.json",
        },
    }


def validate_protocol(payload: dict[str, Any]) -> ExperimentProtocol:
    try:
        return ExperimentProtocol.model_validate(migrate_protocol(payload))
    except ValidationError as exc:
        raise ProtocolError(str(exc)) from exc


def load_protocol(path: Path) -> ExperimentProtocol:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read experiment protocol {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("experiment protocol root must be an object")
    return validate_protocol(payload)


def protocol_sha256(protocol: ExperimentProtocol) -> str:
    return hashlib.sha256(canonical_json_bytes(protocol.model_dump(mode="json"))).hexdigest()
