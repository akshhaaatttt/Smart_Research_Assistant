"""Benchmarking utilities for comparing embedding models and vector stores.

Produces speed/quality/storage measurements as plain rows for the comparison charts,
run over the indexed corpus and a few probe queries for reproducibility.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.embeddings.embedder import EmbeddingManager
from src.retrieval.factory import build_vector_store
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.schemas import Chunk

logger = get_logger(__name__)


def compare_embedding_models(
    chunk_texts: List[str],
    queries: List[str],
    model_keys: List[str],
    config: AppConfig,
) -> List[Dict]:
    """Measure encode time and average top-1 self-similarity for each model."""
    manager = EmbeddingManager(config)
    rows: List[Dict] = []
    for key in model_keys:
        try:
            model = manager.get_model(key)
        except Exception as exc:
            logger.warning("Skipping embedding model '%s': %s", key, exc)
            continue

        start = time.perf_counter()
        chunk_vecs = model.embed_texts(chunk_texts)
        encode_time = time.perf_counter() - start

        query_vecs = model.embed_texts(queries)
        sims = query_vecs @ chunk_vecs.T  # (Q, C)
        avg_top1 = float(np.mean(np.max(sims, axis=1))) if sims.size else float("nan")

        rows.append(
            {
                "model": key,
                "encode_time_s": round(encode_time, 4),
                "avg_top1_similarity": round(avg_top1, 4),
                "dimension": model.dimension,
            }
        )
        logger.info("Benchmarked embedding model '%s': %.3fs, top1=%.3f",
                    key, encode_time, avg_top1)
    return rows


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 4)


def compare_vector_stores(
    chunks: List[Chunk],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    backends: List[str],
    config: AppConfig,
    top_k: int = 4,
    query_repeats: int = 20,
) -> List[Dict]:
    """Measure index time, mean query latency, and on-disk storage per backend."""
    dim = embeddings.shape[1]
    rows: List[Dict] = []
    for backend in backends:
        try:
            store = build_vector_store(dim, backend=backend, config=config)
            store.reset()
        except Exception as exc:
            logger.warning("Skipping backend '%s': %s", backend, exc)
            continue

        start = time.perf_counter()
        store.add(chunks, embeddings)
        store.persist()
        index_time = time.perf_counter() - start

        # Average query latency over several repeats for a stable number.
        start = time.perf_counter()
        for _ in range(query_repeats):
            store.search(query_embedding, top_k)
        query_ms = (time.perf_counter() - start) / query_repeats * 1000.0

        persist_dir = config.resolve_path(f"vector_store.{backend}.persist_dir")
        rows.append(
            {
                "backend": backend,
                "index_time_s": round(index_time, 4),
                "query_time_ms": round(query_ms, 4),
                "storage_mb": _dir_size_mb(Path(persist_dir)),
            }
        )
        logger.info("Benchmarked backend '%s': index=%.3fs, query=%.3fms",
                    backend, index_time, query_ms)
    return rows
