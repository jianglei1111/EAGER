class IndexingError(ValueError):
    """Base error for corpus, model pin, and index contract failures."""


class CorpusValidationError(IndexingError):
    """Raised when a processed corpus violates the canonical index contract."""


class ModelRegistryError(IndexingError):
    """Raised when a model pin or artifact differs from its immutable registry entry."""


class IndexValidationError(IndexingError):
    """Raised when an index artifact differs from its recorded summary."""


__all__ = [
    "CorpusValidationError",
    "IndexValidationError",
    "IndexingError",
    "ModelRegistryError",
]
