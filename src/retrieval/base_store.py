"""Abstract vector-store interface shared by the FAISS and ChromaDB backends.

Scores returned by :meth:`search` are always cosine similarities in ``[-1, 1]``
(higher is better) regardless of backend, so callers never deal with distances.
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np

from src.utils.schemas import Chunk, RetrievedChunk


class BaseVectorStore(abc.ABC):
    """Common interface for vector stores."""

    backend: str = "base"

    def __init__(self, dimension: int, collection_name: str) -> None:
        self.dimension = dimension
        self.collection_name = collection_name

    @abc.abstractmethod
    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Index ``chunks`` with their aligned ``embeddings`` (shape ``(N, dim)``)."""

    @abc.abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks to ``query_embedding``."""

    @abc.abstractmethod
    def count(self) -> int:
        """Number of vectors currently stored."""

    @abc.abstractmethod
    def persist(self) -> None:
        """Persist the index/metadata to disk."""

    @abc.abstractmethod
    def load(self) -> bool:
        """Load a previously persisted index. Returns True if something was loaded."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Delete all stored vectors and metadata (start fresh)."""

    def _validate_add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
                f"length mismatch."
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embeddings must have shape (N, {self.dimension}); got {embeddings.shape}."
            )
