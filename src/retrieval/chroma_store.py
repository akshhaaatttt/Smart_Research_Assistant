"""ChromaDB-backed vector store.

A persistent embedded DB that stores documents, metadata, and vectors together.
Configured for cosine space; Chroma's cosine *distance* is converted back to a
*similarity* so the score contract matches the FAISS store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.retrieval.base_store import BaseVectorStore
from src.utils.exceptions import VectorStoreError
from src.utils.logger import get_logger
from src.utils.schemas import Chunk, RetrievedChunk

logger = get_logger(__name__)

# Chroma metadata values must be scalars; these keys are reserved for our own use.
_RESERVED = {"chunk_id", "doc_id", "source", "chunk_index"}


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma only accepts str/int/float/bool metadata values."""
    clean: Dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            continue
        else:
            clean[key] = str(value)
    return clean


class ChromaVectorStore(BaseVectorStore):
    backend = "chroma"

    def __init__(
        self, dimension: int, collection_name: str, persist_dir: str, distance: str = "cosine"
    ) -> None:
        super().__init__(dimension, collection_name)
        import chromadb

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._distance = distance
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self._validate_add(chunks, embeddings)
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [_sanitize_metadata(c.to_metadata()) for c in chunks]
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings.astype(np.float32).tolist(),
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma add failed: {exc}") from exc
        logger.info("Chroma: added %d vectors (total=%d)", len(chunks), self.count())

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        if self.count() == 0:
            return []
        top_k = min(top_k, self.count())
        try:
            res = self._collection.query(
                query_embeddings=[np.asarray(query_embedding, dtype=np.float32).tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        results: List[RetrievedChunk] = []
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            similarity = 1.0 - float(dist)  # cosine distance -> cosine similarity
            chunk = self._chunk_from_storage(doc, meta)
            results.append(RetrievedChunk(chunk=chunk, score=similarity, rank=rank))
        return results

    @staticmethod
    def _chunk_from_storage(document: str, meta: Dict[str, Any]) -> Chunk:
        extra = {k: v for k, v in meta.items() if k not in _RESERVED}
        return Chunk(
            content=document,
            source=meta.get("source", ""),
            doc_id=meta.get("doc_id", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            metadata=extra,
            chunk_id=meta.get("chunk_id", ""),
        )

    def count(self) -> int:
        return int(self._collection.count())

    def persist(self) -> None:
        # PersistentClient writes through on every mutation; nothing extra to do.
        logger.info("Chroma: persisted (%d vectors) at %s", self.count(), self._persist_dir)

    def load(self) -> bool:
        # The collection is opened (and thus loaded) in __init__.
        loaded = self.count() > 0
        if loaded:
            logger.info("Chroma: loaded %d vectors from %s", self.count(), self._persist_dir)
        return loaded

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # collection may not exist yet
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self._distance},
        )
        logger.info("Chroma: store reset")
