from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from eager import __version__
from eager.answering import OpenAIAnswerGenerator, StaticAnswerGenerator
from eager.config import ConfigResolutionError, canonical_json_bytes, resolve_config
from eager.data import (
    DatasetAdapterError,
    build_scaling_corpora,
    get_dataset_adapter,
    list_registered_splits,
    load_prepared_batches,
    prepare_registered_dataset,
    validate_prepared_dataset,
)
from eager.experiments import (
    AutoPolicyExecutor,
    ExperimentRunError,
    ExperimentRunner,
    FixedPolicyExecutor,
)
from eager.evaluation import GPTAnswerJudge
from eager.indexing import (
    DenseIndexSummary,
    IndexingError,
    build_dense_index,
    build_sqlite_surface_store,
    build_structured_resources,
    build_surface_scope_manifest,
    build_tiny_index,
    default_huggingface_snapshot_path,
    load_corpus_snapshot,
    load_evidence_texts,
    load_model_registry,
    validate_corpus,
    validate_dense_index,
    validate_index,
    validate_sqlite_surface_store,
)
from eager.protocol import ProtocolError, load_protocol, protocol_sha256
from eager.pipeline import OnlineAutoRunner, RetrievalEngine
from eager.retrieval import (
    DenseSemanticIndex,
    EntityIndex,
    LexicalReranker,
    QwenReranker,
    RetrievalKernel,
    SqliteEntityIndex,
    TinySemanticIndex,
    kernel_config_for_variant,
)
from eager.runtime.checkpoint import CheckpointError, CheckpointStore, ProvenanceBinding
from eager.runtime.io import (
    DataFormatError,
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
)
from eager.runtime.llm_client import LLMClientError, client_from_resolved_config
from eager.runtime.paths import PathContractError, RepositoryLayout
from eager.schemas import BudgetContract, GeneratedSignalStep, InferencePayload
from eager.signals import (
    EvidenceTitleAnswerer,
    NoSignalController,
    OpenAIChainGenerator,
    OpenAIIntermediateAnswerer,
    OpenAITerminalAnswerer,
    OpenAIRetrievalQuestionGenerator,
    OpenAIReasoningStateGenerator,
    ReasoningStateController,
    RetrievalQuestionController,
    RuleRouter,
    SingleStepChainGenerator,
    StaticChainGenerator,
    StaticQuestionGenerator,
    SubquestionChainController,
    TraceReplayChainGenerator,
)
from eager_baselines import (
    BaselineRegistryError,
    build_shared_stack_executor,
    load_baseline_registry,
)


def _stdout_json(
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    encoding: str | None = None,
) -> str:
    output_encoding = (encoding or getattr(sys.stdout, "encoding", None) or "").lower()
    normalized_encoding = output_encoding.replace("-", "").replace("_", "")
    return json.dumps(
        payload,
        ensure_ascii=not normalized_encoding.startswith("utf8"),
        indent=indent,
        sort_keys=sort_keys,
    )


def _repo_path(layout: RepositoryLayout, path: Path) -> Path:
    return path if path.is_absolute() else layout.resolve(path)


def _question_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "question indices must be comma-separated integers"
        ) from exc
    if not indices:
        raise argparse.ArgumentTypeError("question indices must not be empty")
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("question indices must be non-negative")
    if indices != tuple(sorted(set(indices))):
        raise argparse.ArgumentTypeError(
            "question indices must be unique and strictly increasing"
        )
    return indices


def _resolved_config(args: argparse.Namespace, layout: RepositoryLayout):
    base = _repo_path(layout, args.base)
    local = None if args.no_local else _repo_path(layout, args.local)
    return resolve_config(
        base_path=base,
        local_path=local,
        repo_root=layout.root,
        cli_entries=args.set_values,
    )


def command_config_resolve(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    resolved = _resolved_config(args, layout)
    payload = {"config": resolved.public_dict(), "metadata": resolved.public_metadata()}
    if args.output is not None:
        atomic_write_json(_repo_path(layout, args.output), payload)
    if args.print_json or args.output is None:
        print(_stdout_json(payload))
    return 0


def command_protocol_validate(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    path = _repo_path(layout, args.path)
    protocol = load_protocol(path)
    payload = {
        "ok": True,
        "path": path.relative_to(layout.root).as_posix(),
        "schema_version": protocol.schema_version,
        "datasets": sorted(protocol.datasets),
        "sha256": protocol_sha256(protocol),
    }
    if args.write_migrated is not None:
        atomic_write_json(_repo_path(layout, args.write_migrated), protocol.model_dump(mode="json"))
    print(_stdout_json(payload))
    return 0


def command_baseline_list(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    path = layout.resolve(args.registry, must_exist=True)
    registry = load_baseline_registry(path, repo_root=layout.root)
    print(
        _stdout_json(
            {
                "ok": True,
                "registry": path.relative_to(layout.root).as_posix(),
                "baselines": [
                    entry.model_dump(mode="json") for entry in registry.baselines
                ],
            }
        )
    )
    return 0


def command_data_list(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    registry_path = layout.resolve(args.registry, must_exist=True)
    splits = list_registered_splits(registry_path, repo_root=layout.root)
    payload = {
        "ok": True,
        "registry": registry_path.relative_to(layout.root).as_posix(),
        "splits": list(splits),
    }
    print(_stdout_json(payload))
    return 0


def command_data_prepare(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    registry_path = layout.resolve(args.registry, must_exist=True)
    output_dir = layout.resolve(args.output)
    anomalies_path = _optional_anomalies_path(args.anomalies, layout)
    summary = prepare_registered_dataset(
        registry_path,
        args.split,
        output_dir,
        repo_root=layout.root,
        anomalies_path=anomalies_path,
        download_missing=args.download_missing,
        download_timeout_seconds=args.download_timeout,
    )
    payload = {"ok": True, **summary.public_dict(repo_root=layout.root)}
    print(_stdout_json(payload))
    return 0


def command_data_validate(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    if args.prepared is not None:
        if args.against_corpus is not None:
            raise DatasetAdapterError("--against-corpus is only valid with --index")
        prepared_dir = layout.resolve(args.prepared, must_exist=True)
        anomalies_path = _optional_anomalies_path(args.anomalies, layout)
        summary = validate_prepared_dataset(
            prepared_dir,
            repo_root=layout.root,
            anomalies_path=anomalies_path,
        )
        payload = {
            "ok": True,
            "kind": "prepared_dataset",
            **summary.public_dict(repo_root=layout.root),
        }
    elif args.corpus is not None:
        if args.anomalies is not None or args.against_corpus is not None:
            raise DatasetAdapterError(
                "--anomalies and --against-corpus are not valid with --corpus"
            )
        corpus_path = layout.resolve(args.corpus, must_exist=True)
        summary = validate_corpus(corpus_path)
        payload = {
            "ok": True,
            "kind": "corpus",
            "corpus": corpus_path.relative_to(layout.root).as_posix(),
            "summary": summary.model_dump(mode="json"),
        }
    else:
        if args.anomalies is not None:
            raise DatasetAdapterError("--anomalies is only valid with --prepared")
        index_dir = layout.resolve(args.index, must_exist=True)
        corpus_path = (
            layout.resolve(args.against_corpus, must_exist=True)
            if args.against_corpus is not None
            else None
        )
        semantic_dir, summary = _validate_semantic_index(
            index_dir,
            corpus=corpus_path,
            model_registry_path=layout.resolve(args.model_registry),
        )
        payload = {
            "ok": True,
            "kind": "index",
            "index_dir": index_dir.relative_to(layout.root).as_posix(),
            "semantic_index_dir": semantic_dir.relative_to(layout.root).as_posix(),
            "validation_scope": "full" if corpus_path is not None else "structural_only",
            "summary": summary.model_dump(mode="json"),
        }
    print(_stdout_json(payload))
    return 0


def _optional_anomalies_path(raw_path: Path | None, layout: RepositoryLayout) -> Path | None:
    if raw_path is not None:
        return layout.resolve(raw_path, must_exist=True)
    default_path = layout.resolve("wsdm/config/dataset_anomalies.json")
    return default_path if default_path.is_file() else None


def _semantic_index_dir(index_dir: Path) -> Path:
    nested = index_dir / "semantic"
    if (nested / "summary.json").is_file():
        return nested
    return index_dir


def _validate_semantic_index(
    index_dir: Path,
    *,
    corpus,
    model_registry_path: Path | None = None,
):
    semantic_dir = _semantic_index_dir(index_dir)
    payload = read_json(semantic_dir / "summary.json")
    if not isinstance(payload, dict):
        raise IndexingError(f"semantic index summary must be an object: {semantic_dir}")
    index_format = payload.get("index_format")
    if index_format == "eager.tiny-hash-v1":
        return semantic_dir, validate_index(semantic_dir, corpus=corpus)
    if index_format == "eager.dense-npy-v1":
        summary = validate_dense_index(semantic_dir, corpus=corpus)
        if model_registry_path is not None:
            expected_model = load_model_registry(model_registry_path).resolve(
                summary.encoder.name,
                revision=summary.encoder.revision,
                pin_sha256=summary.encoder.pin_sha256,
            )
            if summary.encoder != expected_model:
                raise IndexingError("dense index encoder differs from the model registry")
        return semantic_dir, summary
    raise IndexingError(f"unsupported semantic index format: {index_format!r}")


def _load_semantic_index(
    index_dir: Path,
    *,
    corpus,
    model_registry_path: Path | None,
    model_dir: Path | None,
    semantic_device: str,
):
    semantic_dir, summary = _validate_semantic_index(
        index_dir,
        corpus=corpus,
        model_registry_path=model_registry_path,
    )
    if isinstance(summary, DenseIndexSummary):
        snapshot = (
            model_dir.resolve()
            if model_dir is not None
            else default_huggingface_snapshot_path(summary.encoder)
        )
        return (
            semantic_dir,
            summary,
            DenseSemanticIndex.load(
                semantic_dir,
                corpus=corpus,
                model_snapshot=snapshot,
                device=semantic_device,
            ),
        )
    return semantic_dir, summary, TinySemanticIndex.load(semantic_dir, corpus=corpus)


def command_data_index(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    corpus_path = layout.resolve(args.corpus, must_exist=True)
    output_dir = layout.resolve(args.output)
    if output_dir.exists():
        raise IndexingError(
            f"index output already exists: {output_dir.relative_to(layout.root).as_posix()}"
        )

    snapshot = load_corpus_snapshot(corpus_path)
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        ) as temp_name:
            staging_dir = Path(temp_name)
            build_tiny_index(snapshot, staging_dir, dimensions=args.dimensions)
            validate_index(staging_dir, corpus=snapshot)
            staging_dir.replace(output_dir)
    except OSError as exc:
        raise IndexingError(f"cannot publish index {output_dir}: {exc}") from exc
    summary = validate_index(output_dir, corpus=snapshot)
    payload = {
        "ok": True,
        "corpus": corpus_path.relative_to(layout.root).as_posix(),
        "index_dir": output_dir.relative_to(layout.root).as_posix(),
        "validation_scope": "full",
        "summary": summary.model_dump(mode="json"),
    }
    print(_stdout_json(payload))
    return 0


def command_data_dense_index(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    corpus_path = layout.resolve(args.corpus, must_exist=True)
    corpus = load_corpus_snapshot(corpus_path)
    evidence_texts_path = (
        layout.resolve(args.evidence_texts, must_exist=True)
        if args.evidence_texts is not None
        else None
    )
    evidence_texts = (
        load_evidence_texts(
            evidence_texts_path,
            expected_ids={unit.evidence_id for unit in corpus.evidence_units},
        )
        if evidence_texts_path is not None
        else None
    )
    output_dir = layout.resolve(args.output)
    if output_dir.exists():
        raise IndexingError(
            f"dense index output already exists: {output_dir.relative_to(layout.root).as_posix()}"
        )
    registry_path = layout.resolve(args.model_registry, must_exist=True)
    model = load_model_registry(registry_path).resolve(args.model_name)
    model_snapshot = (
        args.model_dir.resolve()
        if args.model_dir is not None
        else default_huggingface_snapshot_path(model)
    )
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        ) as temp_name:
            staging_dir = Path(temp_name)
            build_dense_index(
                corpus_path,
                staging_dir,
                model=model,
                model_snapshot=model_snapshot,
                device=args.semantic_device,
                batch_size=args.batch_size,
                show_progress_bar=not args.no_progress,
                evidence_texts=evidence_texts,
                evidence_texts_source_sha256=(
                    sha256_file(evidence_texts_path)
                    if evidence_texts_path is not None
                    else None
                ),
            )
            validate_dense_index(
                staging_dir,
                corpus=corpus_path,
                expected_model_pin=model,
            )
            staging_dir.replace(output_dir)
    except OSError as exc:
        raise IndexingError(f"cannot publish dense index {output_dir}: {exc}") from exc
    summary = validate_dense_index(
        output_dir,
        corpus=corpus_path,
        expected_model_pin=model,
    )
    print(
        _stdout_json(
            {
                "ok": True,
                "corpus": corpus_path.relative_to(layout.root).as_posix(),
                "index_dir": output_dir.relative_to(layout.root).as_posix(),
                "model_snapshot": str(model_snapshot),
                "summary": summary.model_dump(mode="json"),
            }
        )
    )
    return 0


def _scope_paths(values: Sequence[str], layout: RepositoryLayout) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        scope, separator, raw_path = value.partition("=")
        if not separator or not scope or not raw_path:
            raise IndexingError("--scope must use NAME=CORPUS_PATH")
        if scope in output:
            raise IndexingError(f"duplicate surface store scope: {scope}")
        output[scope] = layout.resolve(Path(raw_path), must_exist=True)
    return output


def command_data_surface_store(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    scopes = _scope_paths(args.scope, layout)
    output_dir = layout.resolve(args.output)
    if output_dir.exists():
        raise IndexingError(
            f"surface store output already exists: {output_dir.relative_to(layout.root).as_posix()}"
        )
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        ) as temp_name:
            staging_dir = Path(temp_name)
            build_sqlite_surface_store(
                scopes,
                staging_dir,
                spacy_model=args.spacy_model,
                batch_size=args.batch_size,
                n_process=args.n_process,
                repo_root=layout.root,
            )
            validate_sqlite_surface_store(staging_dir)
            staging_dir.replace(output_dir)
    except OSError as exc:
        raise IndexingError(f"cannot publish surface store {output_dir}: {exc}") from exc
    summary = validate_sqlite_surface_store(output_dir)
    print(
        _stdout_json(
            {
                "ok": True,
                "output": output_dir.relative_to(layout.root).as_posix(),
                "summary": summary,
            }
        )
    )
    return 0


def command_data_surface_scope(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    store_dir = layout.resolve(args.store, must_exist=True)
    corpus_path = layout.resolve(args.corpus, must_exist=True)
    output_dir = layout.resolve(args.output)
    if output_dir.exists():
        raise IndexingError(
            f"surface scope output already exists: {output_dir.relative_to(layout.root).as_posix()}"
        )
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        ) as temp_name:
            staging_dir = Path(temp_name)
            build_surface_scope_manifest(
                store_dir=store_dir,
                scope=args.scope,
                corpus_path=corpus_path,
                output_dir=staging_dir,
                repo_root=layout.root,
            )
            staging_dir.replace(output_dir)
    except OSError as exc:
        raise IndexingError(f"cannot publish surface scope {output_dir}: {exc}") from exc
    manifest = read_json(output_dir / "store_manifest.json")
    print(
        _stdout_json(
            {
                "ok": True,
                "output": output_dir.relative_to(layout.root).as_posix(),
                "manifest": manifest,
            }
        )
    )
    return 0


def command_data_structured(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    corpus_path = layout.resolve(args.corpus, must_exist=True)
    corpus = load_corpus_snapshot(corpus_path)
    evidence_texts_path = (
        layout.resolve(args.evidence_texts, must_exist=True)
        if args.evidence_texts is not None
        else None
    )
    evidence_texts = (
        load_evidence_texts(
            evidence_texts_path,
            expected_ids={unit.evidence_id for unit in corpus.evidence_units},
        )
        if evidence_texts_path is not None
        else None
    )
    output_dir = layout.resolve(args.output)
    surface_index_dir = layout.resolve(args.surface_index)
    summary = build_structured_resources(
        corpus_path,
        output_dir,
        surface_index_dir=surface_index_dir,
        spacy_model=args.spacy_model,
        batch_size=args.batch_size,
        window_radius=args.window_radius,
        max_surface_hits=args.max_surface_hits,
        max_token_surfaces=args.max_token_surfaces,
        max_entity_sentences=args.max_entity_sentences,
        evidence_texts=evidence_texts,
        evidence_texts_source_sha256=(
            sha256_file(evidence_texts_path)
            if evidence_texts_path is not None
            else None
        ),
    )
    print(
        _stdout_json(
            {
                "ok": True,
                "corpus": corpus_path.relative_to(layout.root).as_posix(),
                "output": output_dir.relative_to(layout.root).as_posix(),
                "surface_index": surface_index_dir.relative_to(layout.root).as_posix(),
                "summary": summary,
            }
        )
    )
    return 0


def command_data_corpus(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    source_path = layout.resolve(args.source, must_exist=True)
    output_path = layout.resolve(args.output)
    if output_path.exists():
        raise IndexingError(f"corpus output already exists: {output_path.relative_to(layout.root)}")
    adapter = get_dataset_adapter(args.dataset, split=args.source_split)
    snapshot = adapter.build_corpus_snapshot(adapter.read_rows(source_path))
    atomic_write_jsonl(
        output_path,
        (
            record.model_dump(mode="json")
            for record in sorted(snapshot.records, key=lambda item: item.paragraph_id)
        ),
    )
    validated = load_corpus_snapshot(output_path)
    summary_path = output_path.with_name(f"{output_path.stem}.summary.json")
    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "source_split": args.source_split,
        "source_path": source_path.relative_to(layout.root).as_posix(),
        "source_sha256": sha256_file(source_path),
        "corpus_path": output_path.relative_to(layout.root).as_posix(),
        "corpus_sha256": sha256_file(output_path),
        "corpus": validated.summary.model_dump(mode="json"),
    }
    atomic_write_json(summary_path, summary)
    print(_stdout_json({"ok": True, **summary}))
    return 0


def command_data_scale(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    base_corpus = layout.resolve(args.base_corpus, must_exist=True)
    inference_path = layout.resolve(args.inference, must_exist=True)
    distractor_corpora = tuple(
        layout.resolve(path, must_exist=True) for path in args.distractor_corpus
    )
    output_root = layout.resolve(args.output_root)
    manifest = build_scaling_corpora(
        base_corpus=base_corpus,
        inference_path=inference_path,
        distractor_corpora=distractor_corpora,
        output_root=output_root,
        scales=args.scale or (1, 2, 4, 8),
        seed=args.seed,
        hard_fraction=args.hard_fraction,
        repo_root=layout.root,
    )
    print(
        _stdout_json(
            {
                "ok": True,
                "output_root": output_root.relative_to(layout.root).as_posix(),
                "manifest": manifest,
            }
        )
    )
    return 0


def _load_entity_index(
    index_dir: Path,
    corpus,
    *,
    max_matches_per_anchor: int = 32,
):
    surface_dirs = (index_dir / "surface", index_dir.parent / "surface")
    for surface_dir in surface_dirs:
        if (surface_dir / "store_manifest.json").is_file():
            return SqliteEntityIndex.from_store_manifest(
                surface_dir / "store_manifest.json",
                corpus=corpus,
                max_matches_per_anchor=max_matches_per_anchor,
            )
        if (surface_dir / "index_summary.json").is_file():
            return EntityIndex.from_surface_index(
                surface_dir,
                corpus=corpus,
                max_matches_per_anchor=max_matches_per_anchor,
            )
    return EntityIndex.from_corpus(
        corpus,
        max_matches_per_anchor=max_matches_per_anchor,
    )


def _load_reranker(
    *,
    kind: str,
    enabled: bool,
    model_registry_path: Path,
    model_name: str,
    model_dir: Path | None,
    device: str,
    batch_size: int,
    max_length: int,
):
    if kind == "lexical":
        metadata = {
            "enabled": enabled,
            "type": "lexical",
            "implementation": "LexicalReranker",
            "model_name": "lexical_overlap_v1",
            "model_id": "eager.lexical_overlap_score",
            "revision": __version__,
            "pin_sha256": None,
            "artifact_manifest_sha256": None,
            "device": "cpu",
            "dtype": "float64",
            "batch_size": None,
            "max_length": None,
        }
        return (LexicalReranker() if enabled else None), metadata
    if kind != "qwen3":
        raise ConfigResolutionError(f"unsupported reranker type: {kind}")

    model_pin = load_model_registry(model_registry_path).resolve(model_name)
    if model_pin.role != "cross_encoder_reranker":
        raise ConfigResolutionError(
            f"registered model {model_pin.name} has role {model_pin.role}, "
            "expected cross_encoder_reranker"
        )
    metadata = {
        "enabled": enabled,
        "type": "qwen3",
        "implementation": "QwenReranker",
        "model_name": model_pin.name,
        "model_id": model_pin.model_id,
        "revision": model_pin.revision,
        "pin_sha256": model_pin.pin_sha256,
        "artifact_manifest_sha256": model_pin.artifact_manifest_sha256,
        "device": device,
        "dtype": None,
        "batch_size": batch_size,
        "max_length": max_length,
    }
    if not enabled:
        return None, metadata

    snapshot = (
        model_dir.resolve()
        if model_dir is not None
        else default_huggingface_snapshot_path(model_pin)
    )
    try:
        reranker = QwenReranker(
            snapshot,
            device=None if device == "auto" else device,
            model_pin=model_pin,
            revision=model_pin.revision,
            max_length=max_length,
            batch_size=batch_size,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise IndexingError(f"cannot load registered Qwen reranker {model_pin.name}: {exc}") from exc
    metadata.update({"device": reranker.device, "dtype": reranker.dtype})
    return reranker, metadata


def command_retrieve(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    corpus_path = layout.resolve(args.corpus, must_exist=True)
    index_dir = layout.resolve(args.index, must_exist=True)
    corpus = load_corpus_snapshot(corpus_path)
    _, _, semantic_index = _load_semantic_index(
        index_dir,
        corpus=corpus,
        model_registry_path=layout.resolve(args.model_registry),
        model_dir=args.model_dir,
        semantic_device=args.semantic_device,
    )
    base_config = kernel_config_for_variant(args.variant)
    kernel_config = type(base_config).model_validate(
        {
            **base_config.model_dump(mode="python"),
            "candidate_limit": args.candidate_limit,
            "paragraph_limit": args.paragraph_limit,
            "rerank_limit": args.rerank_limit,
            "evidence_limit": args.evidence_limit,
        }
    )
    budget = BudgetContract(
        max_iterations=args.max_iterations,
        max_candidates_per_step=args.candidate_limit,
        max_candidates_per_question=args.candidate_limit * args.max_iterations,
        max_parent_units_per_step=args.paragraph_limit,
        max_parent_units_per_question=args.paragraph_limit * args.max_iterations,
        max_rerank_pairs_per_step=args.rerank_limit,
        max_rerank_pairs_per_question=args.rerank_limit * args.max_iterations,
        max_retained_evidence=args.evidence_limit,
        max_llm_calls_per_step=0,
        max_llm_calls_per_question=0,
        max_input_tokens_per_question=0,
        max_output_tokens_per_question=0,
    )
    if args.chain_step:
        chain_steps = []
        for index, text in enumerate(args.chain_step, start=1):
            dependencies = tuple(
                f"step_{number}"
                for number in sorted(
                    {
                        int(value)
                        for value in re.findall(r"#(\d+)", text)
                        if int(value) < index
                    }
                )
            )
            chain_steps.append(
                GeneratedSignalStep(
                    step_id=f"step_{index}",
                    text=text,
                    depends_on=dependencies,
                )
            )
        chain_generator = StaticChainGenerator(chain_steps)
    else:
        chain_generator = SingleStepChainGenerator()

    if args.mode == "no-signal":
        controller = NoSignalController(max_steps=args.max_iterations)
    elif args.mode == "retrieval-question":
        generator = (
            StaticQuestionGenerator((args.question, *args.signal))
            if args.signal
            else None
        )
        controller = RetrievalQuestionController(
            generator=generator,
            max_steps=args.max_iterations,
        )
    elif args.mode == "subquestion-chain":
        controller = SubquestionChainController(
            generator=chain_generator,
            answerer=EvidenceTitleAnswerer(),
        )
    else:
        controller = None
    reranker, reranker_metadata = _load_reranker(
        kind=args.reranker,
        enabled=kernel_config.fine_reranking_enabled,
        model_registry_path=layout.resolve(args.model_registry),
        model_name=args.reranker_model_name,
        model_dir=args.reranker_model_dir,
        device=args.reranker_device,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
    )
    kernel = RetrievalKernel(
        corpus=corpus,
        semantic_index=semantic_index,
        config=kernel_config,
        reranker=reranker,
        entity_index=_load_entity_index(index_dir, corpus),
    )
    inference_payload = InferencePayload(dataset=args.dataset, question=args.question)
    engine = RetrievalEngine(kernel)
    router_payload = None
    if args.mode == "auto":
        routed = OnlineAutoRunner(
            engine=engine,
            router=RuleRouter(),
            controller_factories={
                "retrieval_question": lambda: RetrievalQuestionController(
                    max_steps=args.max_iterations
                ),
                "subquestion_chain": lambda: SubquestionChainController(
                    generator=chain_generator,
                    answerer=EvidenceTitleAnswerer(),
                ),
            },
        ).run(payload=inference_payload, budget=budget)
        result = routed.run
        router_payload = routed.router.model_dump(mode="json")
    else:
        result = engine.run(
            payload=inference_payload,
            controller=controller,
            budget=budget,
        )
    payload = {
        "ok": True,
        "variant": args.variant,
        "reranker": reranker_metadata,
        "router": router_payload,
        "result": result.model_dump(mode="json"),
    }
    print(_stdout_json(payload))
    return 0


def _experiment_retrieval(
    settings,
    engine: RetrievalEngine,
    *,
    client,
    retrieval_question_prompt: str | None,
    reasoning_state_prompt: str | None,
    chain_prompt: str | None,
    intermediate_answer_prompt: str | None,
    terminal_answer_prompt: str | None = None,
    same_evidence_answer_retry_limit: int = 0,
    replay_chain_generator: TraceReplayChainGenerator | None = None,
):
    if settings.signal_mode == "none":
        return FixedPolicyExecutor(
            engine,
            lambda: NoSignalController(max_steps=settings.max_iterations),
        )
    if settings.signal_mode == "retrieval_question":
        if client is None or retrieval_question_prompt is None:
            raise ConfigResolutionError("retrieval-question mode requires an LLM client and prompt")
        return FixedPolicyExecutor(
            engine,
            lambda: RetrievalQuestionController(
                generator=OpenAIRetrievalQuestionGenerator(
                    client=client,
                    system_prompt=retrieval_question_prompt,
                    evidence_window_mode=settings.rq_evidence_window_mode,
                ),
                max_steps=settings.max_iterations,
            ),
        )
    if settings.signal_mode == "reasoning_state":
        if client is None or reasoning_state_prompt is None:
            raise ConfigResolutionError("reasoning-state mode requires an LLM client and prompt")
        return FixedPolicyExecutor(
            engine,
            lambda: ReasoningStateController(
                generator=OpenAIReasoningStateGenerator(
                    client=client,
                    system_prompt=reasoning_state_prompt,
                    evidence_window_mode=settings.rq_evidence_window_mode,
                ),
                max_steps=settings.max_iterations,
            ),
        )
    if settings.signal_mode == "subquestion_chain":
        if client is None or chain_prompt is None or intermediate_answer_prompt is None:
            raise ConfigResolutionError("subquestion-chain mode requires an LLM client and prompts")
        return FixedPolicyExecutor(
            engine,
            lambda: SubquestionChainController(
                generator=(
                    replay_chain_generator
                    or OpenAIChainGenerator(
                        client=client,
                        system_prompt=chain_prompt,
                        review_valid_plan=settings.chain_plan_review_enabled,
                    )
                ),
                answerer=OpenAIIntermediateAnswerer(
                    client=client,
                    system_prompt=intermediate_answer_prompt,
                    same_evidence_retry_limit=same_evidence_answer_retry_limit,
                ),
                final_answerer=(
                    OpenAITerminalAnswerer(
                        client=client,
                        system_prompt=terminal_answer_prompt,
                    )
                    if terminal_answer_prompt is not None
                    else None
                ),
                max_chain_steps=settings.max_chain_steps,
                max_step_retries=settings.chain_break_retry_limit,
            ),
        )
    if settings.router_type != "rule":
        raise ConfigResolutionError(
            "the experiment CLI currently requires retrieval.router_type=rule for auto mode"
        )
    if (
        client is None
        or retrieval_question_prompt is None
        or chain_prompt is None
        or intermediate_answer_prompt is None
    ):
        raise ConfigResolutionError("auto mode requires an LLM client and all signal prompts")
    return AutoPolicyExecutor(
        OnlineAutoRunner(
            engine=engine,
            router=RuleRouter(),
            controller_factories={
                "retrieval_question": lambda: RetrievalQuestionController(
                    generator=OpenAIRetrievalQuestionGenerator(
                        client=client,
                        system_prompt=retrieval_question_prompt,
                        evidence_window_mode=settings.rq_evidence_window_mode,
                    ),
                    max_steps=settings.max_iterations
                ),
                "subquestion_chain": lambda: SubquestionChainController(
                    generator=(
                        replay_chain_generator
                        or OpenAIChainGenerator(
                            client=client,
                            system_prompt=chain_prompt,
                            review_valid_plan=settings.chain_plan_review_enabled,
                        )
                    ),
                    answerer=OpenAIIntermediateAnswerer(
                        client=client,
                        system_prompt=intermediate_answer_prompt,
                        same_evidence_retry_limit=same_evidence_answer_retry_limit,
                    ),
                    final_answerer=(
                        OpenAITerminalAnswerer(
                            client=client,
                            system_prompt=terminal_answer_prompt,
                        )
                        if terminal_answer_prompt is not None
                        else None
                    ),
                    max_chain_steps=settings.max_chain_steps,
                    max_step_retries=settings.chain_break_retry_limit,
                ),
            },
        )
    )


def command_experiment_run(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    resolved = _resolved_config(args, layout)
    prepared_dir = layout.resolve(args.prepared, must_exist=True)
    index_dir = layout.resolve(args.index, must_exist=True)
    run_dir = layout.resolve(args.run_dir)
    prepared_summary = validate_prepared_dataset(
        prepared_dir,
        repo_root=layout.root,
        anomalies_path=_optional_anomalies_path(args.anomalies, layout),
    )
    retrieval_settings = resolved.config.retrieval.effective_for_dataset(
        prepared_summary.dataset
    )
    retrieval_settings = retrieval_settings.effective_for_signal_mode()
    final_answer_scope = args.final_answer_scope
    if final_answer_scope == "auto":
        final_answer_scope = "global"
    terminal_answer_prompt = None
    if final_answer_scope == "last_subquestion" and args.static_answer is None:
        terminal_answer_prompt = layout.resolve(
            args.answer_prompt,
            must_exist=True,
        ).read_text(encoding="utf-8-sig")
    batch = load_prepared_batches(prepared_dir)
    corpus_path = (
        layout.resolve(args.corpus, must_exist=True)
        if args.corpus is not None
        else prepared_dir / "corpus.jsonl"
    )
    corpus = load_corpus_snapshot(corpus_path)
    if args.corpus is not None:
        paragraph_ids = {record.paragraph_id for record in corpus.records}
        sentence_ids = {unit.evidence_id for unit in corpus.evidence_units}
        for annotation in batch.evaluation.by_handle_id.values():
            if (
                set(annotation.supporting_paragraph_ids) - paragraph_ids
                or set(annotation.supporting_sentence_ids) - sentence_ids
            ):
                raise ConfigResolutionError(
                    "override corpus does not preserve prepared evaluation support"
                )
    model_registry_path = resolved.resolve_path(resolved.config.paths.model_registry)
    semantic_dir, index_summary, semantic_index = _load_semantic_index(
        index_dir,
        corpus=corpus,
        model_registry_path=model_registry_path,
        model_dir=args.model_dir,
        semantic_device=args.semantic_device,
    )
    evidence_texts = (
        load_evidence_texts(
            semantic_dir / index_summary.representation.path,
            expected_ids={unit.evidence_id for unit in corpus.evidence_units},
        )
        if isinstance(index_summary, DenseIndexSummary)
        and index_summary.representation is not None
        else None
    )
    kernel_config = retrieval_settings.kernel_config()
    if args.baseline == "dense_topk":
        kernel_config = kernel_config.model_copy(
            update={
                "entity_expansion_enabled": False,
                "hierarchical_refinement_enabled": False,
                "fine_reranking_enabled": False,
            }
        )
    elif args.baseline == "reranked_topk":
        kernel_config = kernel_config.model_copy(
            update={
                "entity_expansion_enabled": False,
                "hierarchical_refinement_enabled": False,
                "fine_reranking_enabled": True,
            }
        )
    reranker_settings = resolved.config.reranker
    reranker, reranker_metadata = _load_reranker(
        kind=reranker_settings.type,
        enabled=(
            kernel_config.fine_reranking_enabled
            and args.baseline not in {"zero_shot", "dense_topk"}
        ),
        model_registry_path=model_registry_path,
        model_name=reranker_settings.model_name,
        model_dir=args.reranker_model_dir,
        device=reranker_settings.device,
        batch_size=reranker_settings.batch_size,
        max_length=reranker_settings.max_length,
    )
    engine = RetrievalEngine(
        RetrievalKernel(
            corpus=corpus,
            semantic_index=semantic_index,
            config=kernel_config,
            reranker=reranker,
            entity_index=_load_entity_index(
                index_dir,
                corpus,
                max_matches_per_anchor=retrieval_settings.entity_match_limit,
            ),
            evidence_texts=evidence_texts,
        )
    )
    protocol_path = resolved.resolve_path(resolved.config.paths.experiment_protocol)
    protocol = load_protocol(protocol_path)
    client = None
    signal_prompt_paths: dict[str, Path] = {}
    signal_prompt_hashes: dict[str, str] = {}
    signal_prompts: dict[str, str] = {}
    chain_plan_traces_path: Path | None = None
    replay_chain_generator = None
    if args.baseline is None and retrieval_settings.signal_mode != "none":
        signal_mode = retrieval_settings.signal_mode
        requested_prompts: dict[str, Path] = {}
        if signal_mode in {"retrieval_question", "auto"}:
            requested_prompts["retrieval_question"] = args.retrieval_question_prompt
        if signal_mode == "reasoning_state":
            requested_prompts["reasoning_state"] = args.reasoning_state_prompt
        if signal_mode in {"subquestion_chain", "auto"}:
            requested_prompts["subquestion_chain"] = args.chain_prompt
            requested_prompts["intermediate_answer"] = args.intermediate_answer_prompt
        for name, configured_path in requested_prompts.items():
            path = layout.resolve(configured_path, must_exist=True)
            signal_prompt_paths[name] = path
            signal_prompt_hashes[name] = sha256_file(path)
            signal_prompts[name] = path.read_text(encoding="utf-8-sig")
        client = client_from_resolved_config(resolved)
        if args.chain_plan_traces is not None:
            if signal_mode not in {"subquestion_chain", "auto"}:
                raise ConfigResolutionError(
                    "--chain-plan-traces requires subquestion-chain or auto mode"
                )
            chain_plan_traces_path = layout.resolve(
                args.chain_plan_traces,
                must_exist=True,
            )
            replay_chain_generator = TraceReplayChainGenerator(
                read_jsonl(chain_plan_traces_path)
            )
    retrieval = (
        build_shared_stack_executor(
            args.baseline,
            engine=engine,
            max_steps=retrieval_settings.max_iterations,
        )
        if args.baseline is not None
        else _experiment_retrieval(
            retrieval_settings,
            engine,
            client=client,
            retrieval_question_prompt=signal_prompts.get("retrieval_question"),
            reasoning_state_prompt=signal_prompts.get("reasoning_state"),
            chain_prompt=signal_prompts.get("subquestion_chain"),
            intermediate_answer_prompt=signal_prompts.get("intermediate_answer"),
            terminal_answer_prompt=terminal_answer_prompt,
            same_evidence_answer_retry_limit=(
                retrieval_settings.same_evidence_answer_retry_limit
            ),
            replay_chain_generator=replay_chain_generator,
        )
    )

    prompt_path: Path | None = None
    if args.static_answer is not None:
        answer_generator = StaticAnswerGenerator(args.static_answer)
        answer_prompt_sha256 = hashlib.sha256(
            f"static-answer\0{args.static_answer}".encode("utf-8")
        ).hexdigest()
        answer_mode = "static_smoke_only"
    else:
        prompt_path = layout.resolve(args.answer_prompt, must_exist=True)
        system_prompt = prompt_path.read_text(encoding="utf-8-sig")
        client = client_from_resolved_config(resolved)
        answer_generator = OpenAIAnswerGenerator(
            client=client,
            system_prompt=system_prompt,
        )
        answer_prompt_sha256 = sha256_file(prompt_path)
        answer_mode = "openai_compatible"

    judge_prompt_path: Path | None = None
    answer_judge = None
    judge_prompt_sha256: str | None = None
    if args.judge:
        judge_prompt_path = layout.resolve(args.judge_prompt, must_exist=True)
        if client is None:
            client = client_from_resolved_config(resolved)
        answer_judge = GPTAnswerJudge(
            client=client,
            system_prompt=judge_prompt_path.read_text(encoding="utf-8-sig"),
        )
        judge_prompt_sha256 = sha256_file(judge_prompt_path)
    prompt_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "signal_prompt_sha256": signal_prompt_hashes,
                "answer_prompt_sha256": answer_prompt_sha256,
                "judge_prompt_sha256": judge_prompt_sha256,
                "chain_plan_traces_sha256": (
                    sha256_file(chain_plan_traces_path)
                    if chain_plan_traces_path is not None
                    else None
                ),
            }
        )
    ).hexdigest()

    run_config = {
        "config": resolved.public_dict(),
        "metadata": resolved.public_metadata(),
        "experiment": {
            "prepared": prepared_dir.relative_to(layout.root).as_posix(),
            "corpus": {
                "path": corpus_path.relative_to(layout.root).as_posix(),
                "sha256": sha256_file(corpus_path),
                "override": args.corpus is not None,
            },
            "index": index_dir.relative_to(layout.root).as_posix(),
            "execution_profile": args.profile,
            "question_workers": args.question_workers,
            "effective_retrieval": retrieval_settings.model_dump(
                mode="json",
                exclude={"dataset_profiles"},
            ),
            "method": args.baseline or "eager",
            "answer_mode": answer_mode,
            "final_answer_scope": final_answer_scope,
            "signal_prompts": {
                name: {
                    "path": path.relative_to(layout.root).as_posix(),
                    "sha256": signal_prompt_hashes[name],
                }
                for name, path in signal_prompt_paths.items()
            },
            "chain_plan_traces": (
                {
                    "path": chain_plan_traces_path.relative_to(layout.root).as_posix(),
                    "sha256": sha256_file(chain_plan_traces_path),
                }
                if chain_plan_traces_path is not None
                else None
            ),
            "answer_prompt": (
                prompt_path.relative_to(layout.root).as_posix()
                if prompt_path is not None
                else None
            ),
            "static_answer_sha256": (
                answer_prompt_sha256 if args.static_answer is not None else None
            ),
            "judge_enabled": args.judge,
            "judge_prompt": (
                judge_prompt_path.relative_to(layout.root).as_posix()
                if judge_prompt_path is not None
                else None
            ),
            "models": {
                "semantic_encoder": index_summary.encoder.model_dump(mode="json"),
                "reranker": reranker_metadata,
            },
            "evidence_text_representation": (
                index_summary.representation.model_dump(mode="json")
                if isinstance(index_summary, DenseIndexSummary)
                and index_summary.representation is not None
                else None
            ),
        },
    }
    if args.question_indices is not None:
        run_config["experiment"]["question_indices"] = list(args.question_indices)
    model_bundle_sha256 = hashlib.sha256(
        canonical_json_bytes(run_config["experiment"]["models"])
    ).hexdigest()
    binding = ProvenanceBinding(
        resolved_config_sha256=hashlib.sha256(
            canonical_json_bytes(run_config)
        ).hexdigest(),
        data_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "prepared_manifest_sha256": prepared_summary.manifest_sha256,
                    "corpus_sha256": sha256_file(corpus_path),
                }
            )
        ).hexdigest(),
        index_sha256=sha256_file(semantic_dir / "summary.json"),
        model_revision=model_bundle_sha256,
        prompt_sha256=prompt_sha256,
    )
    index_metrics = {
        "build_seconds": (
            index_summary.build.build_seconds
            if isinstance(index_summary, DenseIndexSummary)
            else None
        ),
        "bytes": sum(path.stat().st_size for path in index_dir.rglob("*") if path.is_file()),
        "vector_count": index_summary.vectors.record_count,
        "vector_sha256": index_summary.vectors.sha256,
        "dimensions": index_summary.dimensions,
    }
    try:
        result = ExperimentRunner(
            retrieval=retrieval,
            answer_generator=answer_generator,
            protocol=protocol,
            repo_root=layout.root,
            answer_judge=answer_judge,
            final_answer_scope=final_answer_scope,
        ).run(
            run_dir=run_dir,
            dataset=prepared_summary.dataset,
            split=prepared_summary.split,
            profile=args.profile,
            inference=batch.inference,
            evaluation=batch.evaluation,
            budget=retrieval_settings.budget_contract(),
            binding=binding,
            resolved_config=run_config,
            index_metrics=index_metrics,
            resume=not args.no_resume,
            max_questions=args.max_questions,
            question_workers=args.question_workers,
            question_indices=args.question_indices,
        )
    finally:
        if client is not None:
            client.close()
    print(
        _stdout_json(
            {
                "ok": True,
                "run_dir": run_dir.relative_to(layout.root).as_posix(),
                "status": result.status,
                "prediction_count": result.prediction_count,
                "resumed_count": result.resumed_count,
                "metrics": result.metrics,
            }
        )
    )
    return 0


def command_smoke(args: argparse.Namespace, layout: RepositoryLayout) -> int:
    resolved = _resolved_config(args, layout)
    protocol_path = resolved.resolve_path(resolved.config.paths.experiment_protocol)
    protocol = load_protocol(protocol_path)
    inference_path = layout.resolve("wsdm/fixtures/development/inference.json", must_exist=True)
    annotations_path = layout.resolve("wsdm/fixtures/development/annotations.json", must_exist=True)
    inference = read_json(inference_path)
    annotations = read_json(annotations_path)
    inference_handles = {
        str(row["question_handle"])
        for row in inference.get("records", [])
        if isinstance(row, dict) and row.get("question_handle")
    }
    annotation_handles = {
        str(row["question_handle"])
        for row in annotations.get("annotations", [])
        if isinstance(row, dict) and row.get("question_handle")
    }
    if not inference_handles or inference_handles != annotation_handles:
        raise ConfigResolutionError("development fixture inference/annotation handles do not match")

    with tempfile.TemporaryDirectory(prefix="eager-phase1-smoke-") as temp_dir:
        checkpoint_path = Path(temp_dir) / "checkpoints.jsonl"
        store = CheckpointStore(checkpoint_path)
        binding = ProvenanceBinding(
            resolved.public_metadata()["resolved_config_sha256"],
            "fixture-data",
            "fixture-index",
            "mock-model-revision",
            "fixture-prompt",
        )
        store.append(
            checkpoint_id="smoke:0",
            stage="phase1",
            binding=binding,
            state={"question_count": len(inference_handles)},
        )
        resumed = store.resume("smoke:0", binding)
        if resumed is None:
            raise RuntimeError("checkpoint smoke did not resume")

    payload: dict[str, Any] = {
        "ok": True,
        "offline": True,
        "config_sha256": resolved.public_metadata()["resolved_config_sha256"],
        "protocol_sha256": protocol_sha256(protocol),
        "dataset_count": len(protocol.datasets),
        "fixture_question_count": len(inference_handles),
        "checkpoint_resume": True,
    }
    if args.json:
        print(_stdout_json(payload, indent=None))
    else:
        print("Phase 1 offline smoke passed")
        for key, value in payload.items():
            print(f"{key}={value}")
    return 0


def add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base", type=Path, default=Path("configs/wsdm/base.json"))
    parser.add_argument("--local", type=Path, default=Path("configs/llm_config.local.json"))
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--set", dest="set_values", action="append", default=[], metavar="PATH=VALUE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eager", description="EAGER unified system command line interface.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Resolve layered configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    resolve_parser = config_subparsers.add_parser("resolve", help="Resolve base, local, environment, and CLI layers.")
    add_config_options(resolve_parser)
    resolve_parser.add_argument("--output", type=Path)
    resolve_parser.add_argument("--print-json", action="store_true")
    resolve_parser.set_defaults(handler=command_config_resolve)

    protocol_parser = subparsers.add_parser("protocol", help="Validate or migrate experiment protocol files.")
    protocol_subparsers = protocol_parser.add_subparsers(dest="protocol_command", required=True)
    validate_parser = protocol_subparsers.add_parser("validate", help="Validate an experiment protocol.")
    validate_parser.add_argument("--path", type=Path, default=Path("wsdm/config/experiment_protocol.json"))
    validate_parser.add_argument("--write-migrated", type=Path)
    validate_parser.set_defaults(handler=command_protocol_validate)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Inspect registered shared-stack and external baselines.",
    )
    baseline_subparsers = baseline_parser.add_subparsers(
        dest="baseline_command",
        required=True,
    )
    baseline_list_parser = baseline_subparsers.add_parser(
        "list",
        help="Validate and list the baseline registry.",
    )
    baseline_list_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("wsdm/config/baseline_registry.json"),
    )
    baseline_list_parser.set_defaults(handler=command_baseline_list)

    data_parser = subparsers.add_parser(
        "data",
        help="Prepare registered datasets and validate or index canonical corpora.",
    )
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_list_parser = data_subparsers.add_parser("list", help="List validated split registrations.")
    data_list_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("wsdm/config/split_registry.json"),
    )
    data_list_parser.set_defaults(handler=command_data_list)

    data_prepare_parser = data_subparsers.add_parser(
        "prepare",
        help="Prepare isolated question, annotation, and canonical corpus files.",
    )
    data_prepare_parser.add_argument("--split", required=True)
    data_prepare_parser.add_argument("--output", type=Path, required=True)
    data_prepare_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("wsdm/config/split_registry.json"),
    )
    data_prepare_parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Explicitly acquire a missing source from its immutable v3 registry URL.",
    )
    data_prepare_parser.add_argument("--download-timeout", type=float, default=120.0)
    data_prepare_parser.add_argument(
        "--anomalies",
        type=Path,
        help="Optional repo-relative, source-hash-bound dataset anomaly manifest.",
    )
    data_prepare_parser.set_defaults(handler=command_data_prepare)

    data_validate_parser = data_subparsers.add_parser(
        "validate",
        help="Validate prepared data, a canonical corpus, or an index.",
    )
    validation_target = data_validate_parser.add_mutually_exclusive_group(required=True)
    validation_target.add_argument("--prepared", type=Path)
    validation_target.add_argument("--corpus", type=Path)
    validation_target.add_argument("--index", type=Path)
    data_validate_parser.add_argument(
        "--anomalies",
        type=Path,
        help="Optional repo-relative, source-hash-bound dataset anomaly manifest.",
    )
    data_validate_parser.add_argument(
        "--against-corpus",
        type=Path,
        help="Repo-relative canonical corpus used to verify an index exactly.",
    )
    data_validate_parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("configs/wsdm/models/registry.json"),
        help="Model registry used to verify a dense index encoder pin.",
    )
    data_validate_parser.set_defaults(handler=command_data_validate)

    data_index_parser = data_subparsers.add_parser(
        "index",
        help="Build and validate a deterministic CPU fixture index.",
    )
    data_index_parser.add_argument("--corpus", type=Path, required=True)
    data_index_parser.add_argument("--output", type=Path, required=True)
    data_index_parser.add_argument("--dimensions", type=int, default=64)
    data_index_parser.set_defaults(handler=command_data_index)

    data_dense_index_parser = data_subparsers.add_parser(
        "dense-index",
        help="Build a corpus- and model-bound MiniLM float32 semantic index.",
    )
    data_dense_index_parser.add_argument("--corpus", type=Path, required=True)
    data_dense_index_parser.add_argument("--output", type=Path, required=True)
    data_dense_index_parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("configs/wsdm/models/registry.json"),
    )
    data_dense_index_parser.add_argument("--model-name", default="minilm_l6_v2")
    data_dense_index_parser.add_argument("--model-dir", type=Path)
    data_dense_index_parser.add_argument("--semantic-device", default="cuda")
    data_dense_index_parser.add_argument("--batch-size", type=int, default=256)
    data_dense_index_parser.add_argument("--no-progress", action="store_true")
    data_dense_index_parser.add_argument(
        "--evidence-texts",
        type=Path,
        help="Optional evidence_id/text JSONL used as the retrieval text representation.",
    )
    data_dense_index_parser.set_defaults(handler=command_data_dense_index)

    data_surface_store_parser = data_subparsers.add_parser(
        "surface-store",
        help="Build one shared SQLite surface store for nested corpus scopes.",
    )
    data_surface_store_parser.add_argument(
        "--scope",
        action="append",
        required=True,
        metavar="NAME=CORPUS_PATH",
    )
    data_surface_store_parser.add_argument("--output", type=Path, required=True)
    data_surface_store_parser.add_argument("--spacy-model", default="en_core_web_sm")
    data_surface_store_parser.add_argument("--batch-size", type=int, default=128)
    data_surface_store_parser.add_argument("--n-process", type=int, default=1)
    data_surface_store_parser.set_defaults(handler=command_data_surface_store)

    data_surface_scope_parser = data_subparsers.add_parser(
        "surface-scope",
        help="Bind a corpus to one membership scope in a shared SQLite surface store.",
    )
    data_surface_scope_parser.add_argument("--store", type=Path, required=True)
    data_surface_scope_parser.add_argument("--scope", required=True)
    data_surface_scope_parser.add_argument("--corpus", type=Path, required=True)
    data_surface_scope_parser.add_argument("--output", type=Path, required=True)
    data_surface_scope_parser.set_defaults(handler=command_data_surface_scope)

    data_structured_parser = data_subparsers.add_parser(
        "structured",
        help="Build annotation-free sentence, entity, window, and bounded surface resources.",
    )
    data_structured_parser.add_argument("--corpus", type=Path, required=True)
    data_structured_parser.add_argument("--output", type=Path, required=True)
    data_structured_parser.add_argument("--surface-index", type=Path, required=True)
    data_structured_parser.add_argument("--spacy-model", default="en_core_web_sm")
    data_structured_parser.add_argument("--batch-size", type=int, default=128)
    data_structured_parser.add_argument("--window-radius", type=int, default=1)
    data_structured_parser.add_argument("--max-surface-hits", type=int, default=32)
    data_structured_parser.add_argument("--max-token-surfaces", type=int, default=128)
    data_structured_parser.add_argument("--max-entity-sentences", type=int, default=32)
    data_structured_parser.add_argument(
        "--evidence-texts",
        type=Path,
        help="Optional evidence_id/text JSONL used to build resolved surface resources.",
    )
    data_structured_parser.set_defaults(handler=command_data_structured)

    data_corpus_parser = data_subparsers.add_parser(
        "corpus",
        help="Build an annotation-free canonical corpus from a complete raw dataset source.",
    )
    data_corpus_parser.add_argument("--dataset", choices=("hotpotqa", "2wiki", "musique"), required=True)
    data_corpus_parser.add_argument("--source", type=Path, required=True)
    data_corpus_parser.add_argument("--source-split", required=True)
    data_corpus_parser.add_argument("--output", type=Path, required=True)
    data_corpus_parser.set_defaults(handler=command_data_corpus)

    data_scale_parser = data_subparsers.add_parser(
        "scale",
        help="Build deterministic nested 1x/2x/4x/8x shared corpora.",
    )
    data_scale_parser.add_argument("--base-corpus", type=Path, required=True)
    data_scale_parser.add_argument("--inference", type=Path, required=True)
    data_scale_parser.add_argument(
        "--distractor-corpus", type=Path, action="append", required=True
    )
    data_scale_parser.add_argument("--output-root", type=Path, required=True)
    data_scale_parser.add_argument("--scale", type=int, action="append")
    data_scale_parser.add_argument("--seed", type=int, default=20260713)
    data_scale_parser.add_argument("--hard-fraction", type=float, default=0.5)
    data_scale_parser.set_defaults(handler=command_data_scale)

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Run the shared Phase 3 retrieval engine on one annotation-isolated question.",
    )
    retrieve_parser.add_argument("--dataset", required=True)
    retrieve_parser.add_argument("--question", required=True)
    retrieve_parser.add_argument("--corpus", type=Path, required=True)
    retrieve_parser.add_argument("--index", type=Path, required=True)
    retrieve_parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("configs/wsdm/models/registry.json"),
    )
    retrieve_parser.add_argument("--model-dir", type=Path)
    retrieve_parser.add_argument("--semantic-device", default="cuda")
    retrieve_parser.add_argument(
        "--reranker",
        choices=("lexical", "qwen3"),
        default="lexical",
    )
    retrieve_parser.add_argument(
        "--reranker-model-name",
        default="qwen3_reranker_0_6b",
    )
    retrieve_parser.add_argument("--reranker-model-dir", type=Path)
    retrieve_parser.add_argument(
        "--reranker-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    retrieve_parser.add_argument("--reranker-batch-size", type=int, default=4)
    retrieve_parser.add_argument("--reranker-max-length", type=int, default=1024)
    retrieve_parser.add_argument(
        "--mode",
        choices=("retrieval-question", "subquestion-chain", "no-signal", "auto"),
        default="retrieval-question",
    )
    retrieve_parser.add_argument(
        "--variant",
        choices=(
            "full",
            "without_entity_expansion",
            "without_hierarchical_refinement",
        ),
        default="full",
    )
    retrieve_parser.add_argument(
        "--signal",
        action="append",
        default=[],
        help="Optional fixed supplementary retrieval question; repeat for multiple steps.",
    )
    retrieve_parser.add_argument(
        "--chain-step",
        action="append",
        default=[],
        help="Optional chain subquestion; use #1, #2, ... for dependencies.",
    )
    retrieve_parser.add_argument("--max-iterations", type=int, default=2)
    retrieve_parser.add_argument("--candidate-limit", type=int, default=20)
    retrieve_parser.add_argument("--paragraph-limit", type=int, default=5)
    retrieve_parser.add_argument("--rerank-limit", type=int, default=20)
    retrieve_parser.add_argument("--evidence-limit", type=int, default=10)
    retrieve_parser.set_defaults(handler=command_retrieve)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Run a protocol-bound end-to-end experiment.",
    )
    experiment_subparsers = experiment_parser.add_subparsers(
        dest="experiment_command",
        required=True,
    )
    experiment_run_parser = experiment_subparsers.add_parser(
        "run",
        help="Run retrieval, answering, and evaluation from prepared data.",
    )
    add_config_options(experiment_run_parser)
    experiment_run_parser.add_argument("--prepared", type=Path, required=True)
    experiment_run_parser.add_argument(
        "--corpus",
        type=Path,
        help=(
            "Optional retrieval corpus override. The runner verifies that all prepared "
            "evaluation support remains present and binds the corpus SHA into provenance."
        ),
    )
    experiment_run_parser.add_argument("--index", type=Path, required=True)
    experiment_run_parser.add_argument("--model-dir", type=Path)
    experiment_run_parser.add_argument("--semantic-device", default="cuda")
    experiment_run_parser.add_argument("--reranker-model-dir", type=Path)
    experiment_run_parser.add_argument("--run-dir", type=Path, required=True)
    experiment_run_parser.add_argument(
        "--profile",
        default="wsdm_strict",
        choices=("wsdm_strict", "legacy_reproduction"),
    )
    experiment_run_parser.add_argument("--answer-prompt", type=Path, default=Path("prompts/linear_rag_answer_target_slot_system_prompt.txt"))
    experiment_run_parser.add_argument(
        "--final-answer-scope",
        choices=("auto", "global", "last_subquestion"),
        default="auto",
        help=(
            "Final answer context. auto restores the global answer layer; "
            "last_subquestion is an explicit SC diagnostic mode."
        ),
    )
    experiment_run_parser.add_argument(
        "--retrieval-question-prompt",
        type=Path,
        default=Path("prompts/eager_retrieval_question_system_prompt.md"),
    )
    experiment_run_parser.add_argument(
        "--reasoning-state-prompt",
        type=Path,
        default=Path("prompts/eager_reasoning_state_system_prompt.md"),
    )
    experiment_run_parser.add_argument(
        "--chain-prompt",
        type=Path,
        default=Path("prompts/eager_subquestion_chain_v6_simple_system_prompt.md"),
    )
    experiment_run_parser.add_argument(
        "--chain-plan-traces",
        type=Path,
        help=(
            "Replay question/generated_chain pairs from a prior retrieval trace "
            "for controlled diagnostic runs."
        ),
    )
    experiment_run_parser.add_argument(
        "--intermediate-answer-prompt",
        type=Path,
        default=Path("prompts/eager_intermediate_answer_system_prompt.md"),
    )
    experiment_run_parser.add_argument(
        "--static-answer",
        help="Offline smoke-only answer used instead of an API call.",
    )
    question_selection = experiment_run_parser.add_mutually_exclusive_group()
    question_selection.add_argument("--max-questions", type=int)
    question_selection.add_argument(
        "--question-indices",
        type=_question_indices,
        help=(
            "Comma-separated zero-based prepared-split indices for a sparse "
            "diagnostic run."
        ),
    )
    experiment_run_parser.add_argument(
        "--question-workers",
        type=int,
        default=1,
        help="Concurrent questions sharing one loaded encoder and reranker instance.",
    )
    experiment_run_parser.add_argument(
        "--judge",
        action="store_true",
        help="Evaluate GPTAcc with the configured OpenAI-compatible client.",
    )
    experiment_run_parser.add_argument(
        "--judge-prompt",
        type=Path,
        default=Path("prompts/gpt_judge_prompt.txt"),
    )
    experiment_run_parser.add_argument(
        "--baseline",
        choices=("zero_shot", "dense_topk", "reranked_topk", "ircot_style"),
        help="Run a registered local baseline through the same answer/evaluator stack.",
    )
    experiment_run_parser.add_argument("--no-resume", action="store_true")
    experiment_run_parser.add_argument("--anomalies", type=Path)
    experiment_run_parser.set_defaults(handler=command_experiment_run)

    smoke_parser = subparsers.add_parser("smoke", help="Run the tiny offline Phase 1 smoke test.")
    add_config_options(smoke_parser)
    smoke_parser.add_argument("--json", action="store_true")
    smoke_parser.set_defaults(handler=command_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        layout = RepositoryLayout.discover(Path.cwd())
        return int(args.handler(args, layout))
    except (
        CheckpointError,
        BaselineRegistryError,
        ConfigResolutionError,
        DataFormatError,
        DatasetAdapterError,
        FileNotFoundError,
        IndexingError,
        ExperimentRunError,
        LLMClientError,
        PathContractError,
        ProtocolError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
