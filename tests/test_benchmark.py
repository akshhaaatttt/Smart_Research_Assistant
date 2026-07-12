"""Tests for the benchmarking utilities."""

import numpy as np
import pytest

from src.embeddings.embedder import EmbeddingManager
from src.evaluation.benchmark import compare_embedding_models, compare_vector_stores
from src.utils.config import AppConfig
from src.utils.schemas import Chunk

CHUNKS_TEXT = [
    "Employees get 26 weeks of paid maternity leave.",
    "Customer data must be encrypted at rest using AES-256.",
    "The company matches 401(k) contributions up to five percent.",
]
QUERIES = ["maternity leave duration", "data encryption standard"]


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingManager().get_model("minilm")


def test_compare_embedding_models_single(embedder):
    cfg = AppConfig.load()
    rows = compare_embedding_models(CHUNKS_TEXT, QUERIES, ["minilm"], cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "minilm"
    assert row["dimension"] == 384
    assert row["encode_time_s"] > 0
    assert 0.0 <= row["avg_top1_similarity"] <= 1.0


def test_compare_vector_stores(embedder, tmp_path):
    cfg = AppConfig.load().with_overrides(
        {
            "vector_store": {
                "faiss": {"persist_dir": str(tmp_path / "faiss")},
                "chroma": {"persist_dir": str(tmp_path / "chroma")},
            }
        }
    )
    chunks = [Chunk(content=t, source=f"d{i}.txt", doc_id=f"d{i}", chunk_index=0)
              for i, t in enumerate(CHUNKS_TEXT)]
    embeddings = embedder.embed_texts(CHUNKS_TEXT)
    query = embedder.embed_query("encryption at rest")

    rows = compare_vector_stores(chunks, embeddings, query, ["faiss", "chroma"], cfg)
    assert {r["backend"] for r in rows} == {"faiss", "chroma"}
    for r in rows:
        assert r["index_time_s"] >= 0
        assert r["query_time_ms"] >= 0
        assert r["storage_mb"] >= 0
