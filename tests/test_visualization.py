"""Tests for the visualization builders.

Charts are validated structurally: each returns a Plotly Figure with the expected
number of traces / data, and reductions produce the right shape. This keeps the
visual layer honest without rendering pixels.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

import src.visualization as viz
from src.utils.schemas import Chunk, Document, RetrievedChunk
from src.visualization.embedding_viz import _reduce


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def documents():
    return [
        Document(content="word " * 200, source="a.txt", doc_type="txt",
                 metadata={"filename": "a.txt"}),
        Document(content="word " * 500, source="b.pdf", doc_type="pdf",
                 metadata={"filename": "b.pdf", "num_pages": 3}),
    ]


@pytest.fixture
def chunks():
    return [
        Chunk(content="x" * (100 + i * 10), source=f"{'a.txt' if i < 3 else 'b.pdf'}",
              doc_id="d", chunk_index=i, metadata={"filename": "a.txt" if i < 3 else "b.pdf"})
        for i in range(6)
    ]


@pytest.fixture
def retrieved(chunks):
    return [RetrievedChunk(chunk=c, score=0.9 - 0.1 * i, rank=i) for i, c in enumerate(chunks[:4])]


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def test_document_summary_table(documents):
    df = viz.document_summary_table(documents)
    assert list(df.columns) == ["source", "type", "characters", "words", "pages"]
    assert len(df) == 2


def test_corpus_charts_return_figures(documents, chunks):
    for fig in (
        viz.document_length_distribution(documents),
        viz.chunk_length_distribution(chunks),
        viz.chunks_per_document(chunks),
        viz.document_type_breakdown(documents),
    ):
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def test_reduce_pca_shape():
    emb = np.random.RandomState(0).rand(20, 32).astype(np.float32)
    coords = _reduce(emb, "pca", 5)
    assert coords.shape == (20, 2)


def test_reduce_tsne_small_corpus():
    emb = np.random.RandomState(0).rand(6, 16).astype(np.float32)
    coords = _reduce(emb, "tsne", 30)  # perplexity clamped internally
    assert coords.shape == (6, 2)


def test_embedding_scatter_with_query():
    emb = np.random.RandomState(1).rand(10, 8).astype(np.float32)
    labels = ["a"] * 5 + ["b"] * 5
    q = np.random.RandomState(2).rand(8).astype(np.float32)
    fig = viz.embedding_scatter(emb, labels, method="pca", query_embedding=q)
    assert isinstance(fig, go.Figure)
    # One trace per colour group + the query star.
    names = [t.name for t in fig.data]
    assert "query" in names


def test_embedding_scatter_rejects_empty():
    from src.utils.exceptions import RAGError

    with pytest.raises(RAGError):
        viz.embedding_scatter(np.empty((0, 8)), [])


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def test_retrieval_charts(retrieved):
    bar = viz.retrieval_scores_bar(retrieved)
    decay = viz.score_decay_by_rank(retrieved)
    dist = viz.score_distribution([r.score for r in retrieved])
    for fig in (bar, decay, dist):
        assert isinstance(fig, go.Figure)
    assert list(bar.data[0].x) == [r.score for r in retrieved]


# --------------------------------------------------------------------------- #
# Performance / comparison
# --------------------------------------------------------------------------- #
def test_performance_charts():
    assert isinstance(viz.stage_timing_bar({"ingest": 0.1, "embed": 0.5}), go.Figure)
    assert isinstance(viz.latency_breakdown({"retrieval": 0.2, "generation": 1.1}), go.Figure)
    assert isinstance(viz.token_usage_bar({"prompt_tokens": 200, "completion_tokens": 20}), go.Figure)


def test_comparison_charts():
    emb_rows = [
        {"model": "minilm", "encode_time_s": 0.3, "avg_top1_similarity": 0.71, "dimension": 384},
        {"model": "mpnet", "encode_time_s": 0.9, "avg_top1_similarity": 0.78, "dimension": 768},
    ]
    vs_rows = [
        {"backend": "faiss", "index_time_s": 0.05, "query_time_ms": 1.2, "storage_mb": 0.4},
        {"backend": "chroma", "index_time_s": 0.20, "query_time_ms": 3.1, "storage_mb": 1.1},
    ]
    assert isinstance(viz.embedding_model_comparison(emb_rows), go.Figure)
    assert isinstance(viz.vector_store_comparison(vs_rows), go.Figure)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def test_evaluation_charts_handle_nan():
    aggregate = {
        "faithfulness": 0.82, "answer_relevancy": 0.75, "context_precision": 0.9,
        "context_recall": float("nan"), "answer_correctness": float("nan"),
    }
    radar = viz.metrics_radar(aggregate)
    bar = viz.metrics_bar(aggregate)
    assert isinstance(radar, go.Figure) and isinstance(bar, go.Figure)
    # NaN metrics are dropped from the bar chart.
    assert len(bar.data[0].x) == 3


def test_per_sample_heatmap():
    per_sample = [
        {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.9},
        {"faithfulness": 0.6, "answer_relevancy": 0.5, "context_precision": 0.4},
    ]
    fig = viz.per_sample_heatmap(per_sample)
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].z) == 2
