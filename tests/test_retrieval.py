"""Tests for the semantic Retriever (top-k, thresholding, MMR)."""

import pytest

from src.embeddings.embedder import EmbeddingManager
from src.retrieval.factory import build_vector_store
from src.retrieval.retriever import Retriever
from src.utils.config import AppConfig
from src.utils.exceptions import RetrievalError
from src.utils.schemas import Chunk

DOCS = [
    "Employees are entitled to 26 weeks of paid maternity leave.",
    "Maternity leave may begin up to four weeks before the due date.",
    "Annual leave accrues at two days per month.",
    "All customer data must be encrypted at rest using AES-256.",
    "Access to production requires multi-factor authentication.",
    "The company matches 401(k) contributions up to five percent.",
]


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingManager().get_model("minilm")


@pytest.fixture(scope="module")
def retriever(embedder, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retr")
    cfg = AppConfig.load().with_overrides(
        {
            "vector_store": {"active": "faiss", "faiss": {"persist_dir": str(tmp)}},
            # Disable the query-adaptive floor here so raw top-k behaviour is tested
            # deterministically; a dedicated test below exercises the floor.
            "retrieval": {"relative_score_floor": 0.0},
        }
    )
    store = build_vector_store(embedder.dimension, backend="faiss", config=cfg)
    store.reset()
    chunks = [Chunk(content=t, source=f"d{i}.txt", doc_id=f"d{i}", chunk_index=0)
              for i, t in enumerate(DOCS)]
    store.add(chunks, embedder.embed_texts(DOCS))
    return Retriever(embedder, store, cfg)


def test_basic_top_k(retriever):
    results = retriever.retrieve("How long is maternity leave?", top_k=3)
    assert len(results) == 3
    assert "maternity" in results[0].content.lower()
    assert [r.rank for r in results] == [0, 1, 2]


def test_empty_query_raises(retriever):
    with pytest.raises(RetrievalError):
        retriever.retrieve("   ")


def test_score_threshold_filters(retriever):
    # A very high threshold should filter out weak matches.
    strict = retriever.retrieve("maternity leave", top_k=6, score_threshold=0.9)
    lenient = retriever.retrieve("maternity leave", top_k=6, score_threshold=0.0)
    assert len(strict) <= len(lenient)
    assert all(r.score >= 0.9 for r in strict)


def test_mmr_returns_requested_count_and_diversity(retriever):
    # With two near-duplicate maternity sentences, MMR should still fill top_k
    # and tends to include a non-maternity chunk for diversity.
    mmr = retriever.retrieve("leave policy", top_k=3, use_mmr=True)
    assert len(mmr) == 3
    assert len({r.chunk.chunk_id for r in mmr}) == 3  # no duplicates


def test_mmr_vs_plain_ranks_contiguous(retriever):
    res = retriever.retrieve("data security", top_k=4, use_mmr=True)
    assert [r.rank for r in res] == list(range(len(res)))


def test_relative_score_floor_prunes_weak_context(embedder, tmp_path):
    """With a strong top match, weakly-related chunks are dropped adaptively."""
    cfg = AppConfig.load().with_overrides(
        {
            "vector_store": {"active": "faiss", "faiss": {"persist_dir": str(tmp_path)}},
            "retrieval": {"relative_score_floor": 0.5},
        }
    )
    store = build_vector_store(embedder.dimension, backend="faiss", config=cfg)
    store.reset()
    chunks = [Chunk(content=t, source=f"d{i}.txt", doc_id=f"d{i}", chunk_index=0)
              for i, t in enumerate(DOCS)]
    store.add(chunks, embedder.embed_texts(DOCS))
    retr = Retriever(embedder, store, cfg)

    # A specific query with one dominant match should return fewer than top_k.
    strong = retr.retrieve("How many weeks of paid maternity leave?", top_k=6)
    assert len(strong) < 6
    assert "maternity" in strong[0].content.lower()
    # The best result is always retained.
    assert strong[0].score == max(r.score for r in strong)
