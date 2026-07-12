"""FAISS-backed vector store.

Uses ``IndexFlatIP`` for exact inner-product search (== cosine, since embeddings
are L2-normalized upstream) and keeps chunk text/metadata in a parallel list indexed
by FAISS row id. Persists ``<collection>.faiss`` and ``<collection>.meta.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

from src.retrieval.base_store import BaseVectorStore
from src.utils.exceptions import VectorStoreError
from src.utils.logger import get_logger
from src.utils.schemas import Chunk, RetrievedChunk

logger = get_logger(__name__)


class FAISSVectorStore(BaseVectorStore):
    backend = "faiss"

    def __init__(self, dimension: int, collection_name: str, persist_dir: str) -> None:
        super().__init__(dimension, collection_name)
        import faiss

        self._faiss = faiss
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: List[Chunk] = []

    @property
    def _index_path(self) -> Path:
        return self._persist_dir / f"{self.collection_name}.faiss"

    @property
    def _meta_path(self) -> Path:
        return self._persist_dir / f"{self.collection_name}.meta.json"

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self._validate_add(chunks, embeddings)
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._chunks.extend(chunks)
        logger.info("FAISS: added %d vectors (total=%d)", len(chunks), self.count())

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        if self.count() == 0:
            return []
        top_k = min(top_k, self.count())
        query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query dim {query.shape[1]} != index dim {self.dimension}."
            )
        scores, indices = self._index.search(query, top_k)

        results: List[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            results.append(
                RetrievedChunk(chunk=self._chunks[idx], score=float(score), rank=rank)
            )
        return results

    def count(self) -> int:
        return int(self._index.ntotal)

    def persist(self) -> None:
        try:
            self._faiss.write_index(self._index, str(self._index_path))
            payload = [asdict(c) for c in self._chunks]
            with open(self._meta_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
        except Exception as exc:
            raise VectorStoreError(f"Failed to persist FAISS store: {exc}") from exc
        logger.info("FAISS: persisted %d vectors to %s", self.count(), self._persist_dir)

    def load(self) -> bool:
        if not (self._index_path.exists() and self._meta_path.exists()):
            return False
        try:
            self._index = self._faiss.read_index(str(self._index_path))
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._chunks = [Chunk(**c) for c in payload]
        except Exception as exc:
            raise VectorStoreError(f"Failed to load FAISS store: {exc}") from exc
        if self._index.d != self.dimension:
            raise VectorStoreError(
                f"Loaded FAISS index dim {self._index.d} != expected {self.dimension}. "
                f"The embedding model likely changed; rebuild the index."
            )
        logger.info("FAISS: loaded %d vectors from %s", self.count(), self._persist_dir)
        return True

    def reset(self) -> None:
        self._index = self._faiss.IndexFlatIP(self.dimension)
        self._chunks = []
        for p in (self._index_path, self._meta_path):
            if p.exists():
                p.unlink()
        logger.info("FAISS: store reset")
