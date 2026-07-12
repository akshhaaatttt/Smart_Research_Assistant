"""Retrieval visualizations: similarity scores and retrieved chunks."""

from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go

from src.utils.config import AppConfig
from src.utils.schemas import RetrievedChunk
from src.visualization.theme import apply_layout


def _short_label(rc: RetrievedChunk, width: int = 45) -> str:
    meta = rc.chunk.metadata
    name = meta.get("filename") or meta.get("title") or rc.source
    page = f" p{meta['page']}" if "page" in meta else ""
    return f"[{rc.rank + 1}] {name}{page}"


def retrieval_scores_bar(
    results: List[RetrievedChunk], config: Optional[AppConfig] = None
) -> go.Figure:
    """Horizontal bar of similarity score per retrieved chunk (best at top)."""
    labels = [_short_label(rc) for rc in results]
    scores = [rc.score for rc in results]
    previews = [rc.content[:120].replace("\n", " ") + "..." for rc in results]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(color=scores, colorscale="Blues", cmin=0, cmax=1),
            hovertext=previews,
            hoverinfo="text+x",
        )
    )
    fig.update_xaxes(title="Cosine similarity", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return apply_layout(fig, "Retrieved chunks by similarity", config, showlegend=False)


def score_decay_by_rank(
    results: List[RetrievedChunk], config: Optional[AppConfig] = None
) -> go.Figure:
    """Line chart of similarity vs. rank - shows how quickly relevance drops off."""
    ranks = [rc.rank + 1 for rc in results]
    scores = [rc.score for rc in results]
    fig = go.Figure(go.Scatter(x=ranks, y=scores, mode="lines+markers"))
    fig.update_xaxes(title="Rank", dtick=1)
    fig.update_yaxes(title="Cosine similarity", range=[0, 1])
    return apply_layout(fig, "Score decay by rank", config, showlegend=False)


def score_distribution(
    all_scores: List[float], config: Optional[AppConfig] = None
) -> go.Figure:
    """Histogram of similarity scores across many retrievals (calibration view)."""
    fig = go.Figure(go.Histogram(x=all_scores, nbinsx=20))
    fig.update_xaxes(title="Cosine similarity", range=[0, 1])
    fig.update_yaxes(title="Frequency")
    return apply_layout(fig, "Similarity score distribution", config, showlegend=False)
