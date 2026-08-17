from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, MutableMapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

class ConfigResolutionError(ValueError):
    """Raised when layered configuration cannot be resolved safely."""


class SecretInTrackedConfigError(ConfigResolutionError):
    """Raised when a tracked base configuration contains a secret value."""


class PathSetting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    base: Literal["repo", "config"] = "repo"
    allow_outside_base: bool = False

    @field_validator("path")
    @classmethod
    def reject_absolute_machine_path(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError("tracked path settings must be relative")
        if re.match(r"^[A-Za-z]:[\\/]", value):
            raise ValueError("tracked path settings must not contain a drive path")
        return value


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str = Field(default="OPENAI_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_completion_tokens: int = Field(default=1600, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not re.match(r"^https?://", normalized, flags=re.IGNORECASE):
            raise ValueError("base_url must use http or https")
        return normalized


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Literal["development", "legacy_reproduction", "wsdm_strict"] = "development"
    random_seed: int = Field(default=20260713, ge=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class RerankerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    type: Literal["lexical", "qwen3"] = "lexical"
    model_name: str = Field(default="qwen3_reranker_0_6b", min_length=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(default=4, ge=1, le=128)
    max_length: int = Field(default=1024, ge=128, le=8192)


class PathsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_root: PathSetting
    experiment_protocol: PathSetting
    split_registry: PathSetting
    dataset_anomalies: PathSetting = PathSetting(
        path="wsdm/config/dataset_anomalies.json",
        base="repo",
    )
    model_registry: PathSetting = PathSetting(
        path="configs/wsdm/models/registry.json",
        base="repo",
    )
    prepared_data_root: PathSetting = PathSetting(path="data/prepared", base="repo")
    index_root: PathSetting = PathSetting(path="data/indexes", base="repo")


class DatasetRetrievalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    evidence_derived_state_anchors_enabled: bool | None = None
    max_iterations: int | None = Field(default=None, ge=1)
    candidate_limit: int | None = Field(default=None, ge=1)
    candidate_budget: int | None = Field(default=None, ge=1)
    paragraph_limit: int | None = Field(default=None, ge=1)
    paragraph_budget: int | None = Field(default=None, ge=1)
    rerank_limit: int | None = Field(default=None, ge=1)
    final_rerank_limit: int | None = Field(default=None, ge=0)
    rerank_budget: int | None = Field(default=None, ge=1)
    evidence_limit: int | None = Field(default=None, ge=1)
    max_llm_calls_per_question: int | None = Field(default=None, ge=0)
    original_query_weight: float | None = Field(default=None, ge=0.0)
    signal_query_weight: float | None = Field(default=None, ge=0.0)
    diversity_first: bool | None = None
    diversity_paragraph_limit: int | None = Field(default=None, ge=0)
    candidate_union_enabled: bool | None = None
    candidate_union_semantic_limit: int | None = Field(default=None, ge=1)
    same_evidence_answer_retry_limit: int | None = Field(default=None, ge=0, le=1)


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    signal_mode: Literal[
        "retrieval_question",
        "reasoning_state",
        "subquestion_chain",
        "none",
        "auto",
    ] = "retrieval_question"
    router_type: Literal["rule", "llm", "hybrid"] = "rule"
    entity_expansion_enabled: bool = True
    evidence_derived_state_anchors_enabled: bool = True
    hierarchical_refinement_enabled: bool = True
    fine_reranking_enabled: bool = True
    global_fallback_enabled: bool = True
    diversity_first: bool = False
    diversity_paragraph_limit: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=2, ge=1)
    max_chain_steps: int = Field(default=6, ge=1)
    candidate_limit: int = Field(default=20, ge=1)
    pre_score_candidate_limit: int | None = Field(default=None, ge=1)
    candidate_budget: int = Field(default=40, ge=1)
    paragraph_limit: int = Field(default=5, ge=1)
    paragraph_budget: int = Field(default=10, ge=1)
    rerank_limit: int = Field(default=20, ge=1)
    final_rerank_limit: int = Field(default=0, ge=0)
    rerank_budget: int = Field(default=40, ge=1)
    evidence_limit: int = Field(default=10, ge=1)
    step_context_limit: int = Field(default=10, ge=1)
    rq_evidence_window_mode: Literal["oldest", "latest_balanced"] = "oldest"
    max_llm_calls_per_step: int = Field(default=1, ge=0)
    max_llm_calls_per_question: int = Field(default=4, ge=0)
    max_input_tokens_per_question: int = Field(default=100000, ge=0)
    max_output_tokens_per_question: int = Field(default=10000, ge=0)
    chain_break_retry_limit: int = Field(default=1, ge=0, le=1)
    chain_plan_review_enabled: bool = False
    original_query_weight: float = Field(default=0.4, ge=0.0)
    signal_query_weight: float = Field(default=0.6, ge=0.0)
    entity_match_limit: int = Field(default=32, ge=1, le=256)
    retained_anchor_scope: Literal["all", "latest_step"] = "all"
    retained_text_anchor_limit: int | None = Field(default=None, ge=0)
    candidate_union_enabled: bool = False
    candidate_union_semantic_limit: int = Field(default=20, ge=1)
    same_evidence_answer_retry_limit: int = Field(default=0, ge=0, le=1)
    dataset_profiles: dict[
        Literal["hotpotqa", "2wiki", "musique"],
        DatasetRetrievalProfile,
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_budgets(self) -> "RetrievalSettings":
        if self.candidate_budget < self.candidate_limit:
            raise ValueError("candidate_budget must cover one candidate_limit step")
        if (
            self.pre_score_candidate_limit is not None
            and self.pre_score_candidate_limit < self.candidate_limit
        ):
            raise ValueError("pre_score_candidate_limit cannot be below candidate_limit")
        if self.paragraph_budget < self.paragraph_limit:
            raise ValueError("paragraph_budget must cover one paragraph_limit step")
        if self.rerank_budget < self.rerank_limit:
            raise ValueError("rerank_budget must cover one rerank_limit step")
        return self

    def effective_for_dataset(self, dataset: str) -> "RetrievalSettings":
        profile = self.dataset_profiles.get(dataset)
        if profile is None:
            return self
        payload = self.model_dump(mode="python")
        payload.update(profile.model_dump(mode="python", exclude_none=True))
        return type(self).model_validate(payload)

    def effective_for_signal_mode(self) -> "RetrievalSettings":
        """Apply the chain's local-retrieval policy without changing RQ.

        The legacy chain retrieved each materialized subquestion directly. The
        strict Planner remains question-only, but SC retrieval should not blend
        the full question and every historical anchor into every step.
        """
        if self.signal_mode != "subquestion_chain":
            return self
        return self.model_copy(
            update={
                "evidence_derived_state_anchors_enabled": False,
                "original_query_weight": 0.0,
                "signal_query_weight": 1.0,
                "retained_anchor_scope": "latest_step",
            }
        )

    def kernel_config(self):
        from eager.retrieval import RetrievalKernelConfig

        return RetrievalKernelConfig(
            entity_expansion_enabled=self.entity_expansion_enabled,
            evidence_derived_state_anchors_enabled=self.evidence_derived_state_anchors_enabled,
            hierarchical_refinement_enabled=self.hierarchical_refinement_enabled,
            fine_reranking_enabled=self.fine_reranking_enabled,
            global_fallback_enabled=self.global_fallback_enabled,
            diversity_first=self.diversity_first,
            diversity_paragraph_limit=self.diversity_paragraph_limit,
            candidate_limit=self.candidate_limit,
            pre_score_candidate_limit=self.pre_score_candidate_limit,
            paragraph_limit=self.paragraph_limit,
            rerank_limit=self.rerank_limit,
            final_rerank_limit=self.final_rerank_limit,
            evidence_limit=self.evidence_limit,
            step_context_limit=self.step_context_limit,
            original_query_weight=self.original_query_weight,
            signal_query_weight=self.signal_query_weight,
            retained_anchor_scope=self.retained_anchor_scope,
            retained_text_anchor_limit=self.retained_text_anchor_limit,
            candidate_union_enabled=self.candidate_union_enabled,
            candidate_union_semantic_limit=self.candidate_union_semantic_limit,
        )

    def budget_contract(self):
        from eager.schemas import BudgetContract

        planned_steps = (
            max(self.max_iterations, self.max_chain_steps)
            if self.signal_mode in {"subquestion_chain", "auto"}
            else self.max_iterations
        )
        retry_steps = (
            self.chain_break_retry_limit
            if self.signal_mode in {"subquestion_chain", "auto"}
            else 0
        )
        same_evidence_retry_steps = (
            max(0, planned_steps - 1) * self.same_evidence_answer_retry_limit
            if self.signal_mode in {"subquestion_chain", "auto"}
            else 0
        )
        extra_chain_steps = planned_steps - self.max_iterations
        max_llm_calls_per_question = self.max_llm_calls_per_question
        if self.signal_mode in {"subquestion_chain", "auto"}:
            # Two planner attempts, one answer per non-leaf node, one optional
            # chain retry, two final-answer attempts, and one optional judge.
            chain_llm_floor = planned_steps + retry_steps + same_evidence_retry_steps + 4
            max_llm_calls_per_question = max(
                max_llm_calls_per_question,
                chain_llm_floor,
            )
        return BudgetContract(
            max_iterations=planned_steps + retry_steps,
            max_candidates_per_step=self.candidate_limit,
            max_candidates_per_question=(
                self.candidate_budget
                + (extra_chain_steps + retry_steps) * self.candidate_limit
            ),
            max_parent_units_per_step=self.paragraph_limit,
            max_parent_units_per_question=(
                self.paragraph_budget
                + (extra_chain_steps + retry_steps) * self.paragraph_limit
            ),
            max_rerank_pairs_per_step=self.rerank_limit,
            max_rerank_pairs_per_question=(
                self.rerank_budget
                + (extra_chain_steps + retry_steps) * self.rerank_limit
            ),
            max_retained_evidence=self.evidence_limit,
            max_llm_calls_per_step=self.max_llm_calls_per_step,
            max_llm_calls_per_question=max_llm_calls_per_question,
            max_input_tokens_per_question=self.max_input_tokens_per_question,
            max_output_tokens_per_question=self.max_output_tokens_per_question,
        )


class EagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    llm: LLMSettings
    paths: PathsSettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)


@dataclass(frozen=True)
class ConfigSource:
    kind: Literal["base", "local"]
    path: Path
    sha256: str

    def public_dict(self, repo_root: Path) -> dict[str, str]:
        try:
            display = self.path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            display = "<external-config>"
        return {"kind": self.kind, "path": display, "sha256": self.sha256}


@dataclass(frozen=True)
class ResolvedConfig:
    config: EagerConfig
    repo_root: Path
    base_config_path: Path
    sources: tuple[ConfigSource, ...]
    _secrets: Mapping[str, str]
    environment_override_names: tuple[str, ...] = ()
    cli_override_paths: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        payload = self.config.model_dump(mode="json")
        assert_no_secret_fields(payload)
        return payload

    def public_metadata(self) -> dict[str, Any]:
        payload = self.public_dict()
        encoded = canonical_json_bytes(payload)
        return {
            "schema_version": 1,
            "resolved_config_sha256": hashlib.sha256(encoded).hexdigest(),
            "sources": [source.public_dict(self.repo_root) for source in self.sources],
            "environment_override_names": list(self.environment_override_names),
            "cli_override_paths": list(self.cli_override_paths),
            "private_override_paths": sorted(self._secrets),
        }

    def secret(self, dotted_path: str) -> str | None:
        return self._secrets.get(dotted_path)

    def resolve_path(self, setting: PathSetting) -> Path:
        base = self.repo_root if setting.base == "repo" else self.base_config_path.parent
        candidate = (base / setting.path).resolve()
        if not setting.allow_outside_base:
            try:
                candidate.relative_to(base.resolve())
            except ValueError as exc:
                raise ConfigResolutionError(
                    f"configured path escapes its {setting.base} base: {setting.path}"
                ) from exc
        return candidate


SECRET_FIELD_PATTERN = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?key|token|secret|password|authorization|credential)(?:$|[_-])"
)
SECRET_REFERENCE_SUFFIXES = ("_env", "_environment", "_file", "_path", "_name")


def is_secret_field(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized.endswith(SECRET_REFERENCE_SUFFIXES):
        return False
    return SECRET_FIELD_PATTERN.search(normalized) is not None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigResolutionError(f"cannot read JSON config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigResolutionError(f"config root must be an object: {path}")
    return payload


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(dict(output[key]), value)
        else:
            output[key] = value
    return output


def _decode_scalar(value: str) -> Any:
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _assign_nested(target: MutableMapping[str, Any], parts: list[str], value: Any) -> None:
    if not parts or any(not part for part in parts):
        raise ConfigResolutionError("override path contains an empty segment")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.setdefault(part, {})
        if not isinstance(existing, MutableMapping):
            raise ConfigResolutionError(f"override path crosses a scalar field: {'.'.join(parts)}")
        cursor = existing
    cursor[parts[-1]] = value


def environment_overrides(
    environment: Mapping[str, str] | None = None,
    *,
    prefix: str = "EAGER__",
) -> tuple[dict[str, Any], tuple[str, ...]]:
    values = environment if environment is not None else os.environ
    payload: dict[str, Any] = {}
    names: list[str] = []
    for name in sorted(values):
        if not name.startswith(prefix):
            continue
        parts = [segment.lower() for segment in name[len(prefix) :].split("__")]
        _assign_nested(payload, parts, _decode_scalar(values[name]))
        names.append(name)
    return payload, tuple(names)


def cli_overrides(entries: Iterable[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ConfigResolutionError(f"CLI override must be dotted.path=value: {entry}")
        raw_path, raw_value = entry.split("=", 1)
        _assign_nested(payload, raw_path.strip().split("."), _decode_scalar(raw_value))
    return payload


def _extract_secrets(value: Any, *, prefix: str = "") -> tuple[Any, dict[str, str]]:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        secrets: dict[str, str] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            dotted = f"{prefix}.{key}" if prefix else key
            if is_secret_field(key):
                if child not in (None, ""):
                    secrets[dotted] = str(child)
                continue
            clean_child, nested = _extract_secrets(child, prefix=dotted)
            clean[key] = clean_child
            secrets.update(nested)
        return clean, secrets
    if isinstance(value, list):
        clean_items: list[Any] = []
        secrets: dict[str, str] = {}
        for index, child in enumerate(value):
            clean_child, nested = _extract_secrets(child, prefix=f"{prefix}[{index}]")
            clean_items.append(clean_child)
            secrets.update(nested)
        return clean_items, secrets
    return value, {}


def assert_no_secret_fields(value: Any, *, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            dotted = f"{prefix}.{key}" if prefix else key
            if is_secret_field(key):
                raise ConfigResolutionError(f"secret field cannot enter serialized config: {dotted}")
            assert_no_secret_fields(child, prefix=dotted)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_secret_fields(child, prefix=f"{prefix}[{index}]")


def resolve_config(
    *,
    base_path: Path,
    repo_root: Path,
    local_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    cli_entries: Iterable[str] = (),
) -> ResolvedConfig:
    cli_entries = tuple(cli_entries)
    base_path = base_path.resolve()
    repo_root = repo_root.resolve()
    base_raw = _load_object(base_path)
    base_clean, base_secrets = _extract_secrets(base_raw)
    if base_secrets:
        raise SecretInTrackedConfigError(
            "tracked base configuration contains secret fields: " + ", ".join(sorted(base_secrets))
        )

    merged: dict[str, Any] = dict(base_clean)
    sources = [ConfigSource(kind="base", path=base_path, sha256=sha256_file(base_path))]
    secrets: dict[str, str] = {}

    if local_path is not None and local_path.exists():
        local_path = local_path.resolve()
        local_payload = _load_object(local_path)
        legacy_llm_fields = {
            "provider",
            "base_url",
            "model",
            "api_key",
            "temperature",
            "max_completion_tokens",
            "timeout",
            "timeout_seconds",
            "max_retries",
            "retry_backoff_seconds",
        }
        if "llm" not in local_payload and any(key in local_payload for key in legacy_llm_fields):
            llm_payload = {key: value for key, value in local_payload.items() if key in legacy_llm_fields}
            if "timeout" in llm_payload and "timeout_seconds" not in llm_payload:
                llm_payload["timeout_seconds"] = llm_payload.pop("timeout")
            local_payload = {
                **{key: value for key, value in local_payload.items() if key not in legacy_llm_fields},
                "llm": llm_payload,
            }
        local_clean, local_secrets = _extract_secrets(local_payload)
        merged = _deep_merge(merged, local_clean)
        secrets.update(local_secrets)
        sources.append(ConfigSource(kind="local", path=local_path, sha256=sha256_file(local_path)))

    env_raw, env_names = environment_overrides(environment)
    env_clean, env_secrets = _extract_secrets(env_raw)
    merged = _deep_merge(merged, env_clean)
    secrets.update(env_secrets)

    cli_raw = cli_overrides(cli_entries)
    cli_clean, cli_secrets = _extract_secrets(cli_raw)
    merged = _deep_merge(merged, cli_clean)
    secrets.update(cli_secrets)

    try:
        model = EagerConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigResolutionError(str(exc)) from exc

    resolved = ResolvedConfig(
        config=model,
        repo_root=repo_root,
        base_config_path=base_path,
        sources=tuple(sources),
        _secrets=dict(secrets),
        environment_override_names=env_names,
        cli_override_paths=tuple(entry.split("=", 1)[0].strip() for entry in cli_entries),
    )
    assert_no_secret_fields(resolved.public_dict())
    return resolved
