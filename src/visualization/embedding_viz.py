"""Embedding-space visualizations via dimensionality reduction (PCA / t-SNE)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import AppConfig
from src.utils.exceptions import RAGError
from src.visualization.theme import apply_layout


def _reduce(embeddings: np.ndarray, method: str, perplexity: int) -> np.ndarray:
    """Project embeddings to 2D using PCA or t-SNE."""
    method = method.lower()
    n = embeddings.shape[0]
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    if method == "tsne":
        from sklearn.manifold import TSNE

        # Perplexity must be < n_samples; clamp for small corpora.
        perplexity = max(2, min(perplexity, n - 1))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        ).fit_transform(embeddings)
    raise RAGError(f"Unknown projection method '{method}'. Use 'pca' or 'tsne'.")


def embedding_scatter(
    embeddings: np.ndarray,
    labels: List[str],
    method: Optional[str] = None,
    hover_texts: Optional[List[str]] = None,
    query_embedding: Optional[np.ndarray] = None,
    config: Optional[AppConfig] = None,
) -> go.Figure:
    """2D scatter of chunk embeddings, coloured by ``labels``; an optional query point is projected alongside."""
    config = config or AppConfig.load()
    method = method or config.get("visualization.projection_method", "pca")
    perplexity = config.get("visualization.tsne_perplexity", 30)
    max_points = config.get("visualization.max_points", 500)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise RAGError("embedding_scatter requires a non-empty (N, dim) array.")

    # Subsample for responsiveness on large corpora.
    if embeddings.shape[0] > max_points:
        idx = np.random.RandomState(42).choice(embeddings.shape[0], max_points, replace=False)
        embeddings = embeddings[idx]
        labels = [labels[i] for i in idx]
        if hover_texts:
            hover_texts = [hover_texts[i] for i in idx]

    # Project corpus (+ query) together for a consistent frame.
    stacked = embeddings
    has_query = query_embedding is not None
    if has_query:
        stacked = np.vstack([embeddings, query_embedding.reshape(1, -1)])

    coords = _reduce(stacked, method, perplexity)
    corpus_coords = coords[:-1] if has_query else coords

    fig = px.scatter(
        x=corpus_coords[:, 0],
        y=corpus_coords[:, 1],
        color=labels,
        hover_name=hover_texts if hover_texts else None,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.75))

    if has_query:
        q = coords[-1]
        fig.add_trace(
            go.Scatter(
                x=[q[0]], y=[q[1]], mode="markers",
                marker=dict(size=18, color="black", symbol="star"),
                name="query",
            )
        )

    fig.update_xaxes(title=f"{method.upper()} dim 1")
    fig.update_yaxes(title=f"{method.upper()} dim 2")
    return apply_layout(
        fig, f"Embedding space ({method.upper()} projection)", config,
        legend_title="source",
    )
