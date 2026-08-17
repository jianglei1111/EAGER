class RetrievalError(ValueError):
    """Base error for the shared retrieval kernel."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when a retrieval variant cannot honor the declared budgets."""


__all__ = ["RetrievalConfigurationError", "RetrievalError"]
