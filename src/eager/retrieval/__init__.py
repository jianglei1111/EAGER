"""Shared entity-anchored retrieval kernel."""

from eager.retrieval.anchors import (
    anchors_from_titles,
    extract_query_anchors,
    generate_surface_keys,
    merge_anchors,
    normalize_surface,
)
from eager.retrieval.coarse_ranker import lexical_overlap_score, score_candidates
from eager.retrieval.contracts import (
    RetrievalKernelConfig,
    RetrievalStepResult,
    RetrievalVariant,
    kernel_config_for_variant,
)
from eager.retrieval.entity_mapping import EntityIndex, EntityMapper, EntityMatch, SqliteEntityIndex
from eager.retrieval.errors import RetrievalConfigurationError, RetrievalError
from eager.retrieval.hierarchical_refiner import RefinementResult, refine_candidates
from eager.retrieval.kernel import RetrievalKernel
from eager.retrieval.reranker import LexicalReranker, PairReranker, QwenReranker
from eager.retrieval.semantic import DenseSemanticIndex, SemanticIndex, TinySemanticIndex

__all__ = [
    "EntityIndex",
    "EntityMapper",
    "EntityMatch",
    "DenseSemanticIndex",
    "LexicalReranker",
    "PairReranker",
    "QwenReranker",
    "RefinementResult",
    "RetrievalConfigurationError",
    "RetrievalError",
    "RetrievalKernel",
    "RetrievalKernelConfig",
    "RetrievalStepResult",
    "RetrievalVariant",
    "SemanticIndex",
    "SqliteEntityIndex",
    "TinySemanticIndex",
    "anchors_from_titles",
    "extract_query_anchors",
    "generate_surface_keys",
    "kernel_config_for_variant",
    "lexical_overlap_score",
    "merge_anchors",
    "normalize_surface",
    "refine_candidates",
    "score_candidates",
]
