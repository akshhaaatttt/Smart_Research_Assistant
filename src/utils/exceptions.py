"""Exception hierarchy. Catch a specific error, or ``RAGError`` at app boundaries."""

from __future__ import annotations


class RAGError(Exception):
    """Base class for every error raised by this project."""


class ConfigError(RAGError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class IngestionError(RAGError):
    """Raised when a document cannot be loaded or parsed."""


class UnsupportedFormatError(IngestionError):
    """Raised when a file/URL type has no registered loader."""


class PreprocessingError(RAGError):
    """Raised when text cleaning or normalization fails."""


class ChunkingError(RAGError):
    """Raised when a document cannot be split into chunks."""


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGError):
    """Raised for indexing / persistence / retrieval failures in a vector store."""


class RetrievalError(RAGError):
    """Raised when the retriever cannot return results."""


class GenerationError(RAGError):
    """Raised when the LLM backend fails to produce an answer."""


class EvaluationError(RAGError):
    """Raised when the evaluation pipeline fails."""
