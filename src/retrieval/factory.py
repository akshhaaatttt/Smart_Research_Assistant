"""Vector-store factory: build the configured backend by name.

Backends are imported lazily so installing only one of FAISS / ChromaDB suffices.
"""

from __future__ import annotations

from typing import Optional

from src.retrieval.base_store import BaseVectorStore
from src.utils.config import AppConfig
from src.utils.exceptions import VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_vector_store(
    dimension: int,
    backend: Optional[str] = None,
    config: Optional[AppConfig] = None,
) -> BaseVectorStore:
    """Construct a vector store; ``backend`` defaults to ``vector_store.active``."""
    config = config or AppConfig.load()
    backend = (backend or config.get("vector_store.active")).lower()
    collection = config.get("vector_store.collection_name")

    if backend == "faiss":
        from src.retrieval.faiss_store import FAISSVectorStore

        persist_dir = str(config.resolve_path("vector_store.faiss.persist_dir"))
        store = FAISSVectorStore(dimension, collection, persist_dir)
    elif backend == "chroma":
        from src.retrieval.chroma_store import ChromaVectorStore

        persist_dir = str(config.resolve_path("vector_store.chroma.persist_dir"))
        distance = config.get("vector_store.chroma.distance", "cosine")
        store = ChromaVectorStore(dimension, collection, persist_dir, distance)
    else:
        raise VectorStoreError(
            f"Unknown vector-store backend '{backend}'. Choose 'faiss' or 'chroma'."
        )

    logger.info("Built '%s' vector store (dim=%d, collection=%s)", backend, dimension, collection)
    return store
