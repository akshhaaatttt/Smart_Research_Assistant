"""Tests for the FAISS and ChromaDB vector stores.

The same test body runs against both backends (parametrized) to guarantee they are
truly interchangeable and honour the shared score contract.
"""

import numpy as np
import pytest

from src.embeddings.embedder import EmbeddingManager
from src.retrieval.factory import build_vector_store
from src.utils.config import AppConfig
from src.utils.schemas import Chunk

BACKENDS = ["faiss", "chroma"]

DOCS = [
    "Employees are entitled to 26 weeks of paid maternity leave.",
    "Annual leave accrues at two days per month, up to 24 days per year.",
    "All customer data must be encrypted at rest using AES-256.",
    "Access to production systems requires multi-factor authentication.",
    "The 401(k) plan is matched by the company up to five percent.",
]


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingManager().get_model("minilm")


@pytest.fixture(scope="module")
def chunks():
    return [
        Chunk(content=text, source=f"doc{i}.txt", doc_id=f"d{i}", chunk_index=0,
              metadata={"page": i + 1})
        for i, text in enumerate(DOCS)
    ]


@pytest.fixture(scope="module")
def embeddings(embedder):
    return embedder.embed_texts(DOCS)


def _store(backend, dim, tmp_path):
    cfg = AppConfig.load().with_overrides(
        {
            "vector_store": {
                "active": backend,
                "collection_name": "test_collection",
                "faiss": {"persist_dir": str(tmp_path / "faiss")},
                "chroma": {"persist_dir": str(tmp_path / "chroma")},
            }
        }
    )
    return build_vector_store(dim, backend=backend, config=cfg)


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_and_count(backend, chunks, embeddings, embedder, tmp_path):
    store = _store(backend, embedder.dimension, tmp_path)
    store.reset()
    store.add(chunks, embeddings)
    assert store.count() == len(chunks)


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_returns_relevant_chunk_first(backend, chunks, embeddings, embedder, tmp_path):
    store = _store(backend, embedder.dimension, tmp_path)
    store.reset()
    store.add(chunks, embeddings)

    q = embedder.embed_query("How many weeks of maternity leave do employees get?")
    results = store.search(q, top_k=3)
    assert len(results) == 3
    assert "maternity" in results[0].content.lower()
    # Scores are cosine similarities, sorted descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert -1.0001 <= scores[0] <= 1.0001
    # Ranks are contiguous from 0.
    assert [r.rank for r in results] == [0, 1, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_metadata_preserved(backend, chunks, embeddings, embedder, tmp_path):
    store = _store(backend, embedder.dimension, tmp_path)
    store.reset()
    store.add(chunks, embeddings)
    q = embedder.embed_query("encryption at rest")
    top = store.search(q, top_k=1)[0]
    assert top.chunk.source.startswith("doc")
    assert "page" in top.chunk.metadata


@pytest.mark.parametrize("backend", BACKENDS)
def test_persist_and_reload(backend, chunks, embeddings, embedder, tmp_path):
    store = _store(backend, embedder.dimension, tmp_path)
    store.reset()
    store.add(chunks, embeddings)
    store.persist()

    # A fresh store pointed at the same directory should recover the data.
    reloaded = _store(backend, embedder.dimension, tmp_path)
    assert reloaded.load() is True
    assert reloaded.count() == len(chunks)
    q = embedder.embed_query("multi-factor authentication")
    assert "authentication" in reloaded.search(q, top_k=1)[0].content.lower()


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_empty_store(backend, embedder, tmp_path):
    store = _store(backend, embedder.dimension, tmp_path)
    store.reset()
    q = embedder.embed_query("anything")
    assert store.search(q, top_k=5) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_reset_clears(backend, chunks, embeddings, embedder, tmp_path):
    store = _store(backend, embedder.dimension, tmp_path)
    store.add(chunks, embeddings)
    store.reset()
    assert store.count() == 0


def test_backends_agree_on_top_result(chunks, embeddings, embedder, tmp_path):
    """FAISS and Chroma should retrieve the same top chunk for a query."""
    faiss_store = _store("faiss", embedder.dimension, tmp_path / "a")
    chroma_store = _store("chroma", embedder.dimension, tmp_path / "b")
    for s in (faiss_store, chroma_store):
        s.reset()
        s.add(chunks, embeddings)
    q = embedder.embed_query("company retirement plan matching")
    assert faiss_store.search(q, 1)[0].chunk.content == chroma_store.search(q, 1)[0].chunk.content
