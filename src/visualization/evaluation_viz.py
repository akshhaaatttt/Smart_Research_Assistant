"""Evaluation visualizations: radar, bar, and per-sample heatmap of RAG quality metrics."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import plotly.graph_objects as go

from src.utils.config import AppConfig
from src.visualization.theme import PALETTE, apply_layout

_METRIC_ORDER = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def _pretty(name: str) -> str:
    return name.replace("_", " ").title()


def _present_metrics(aggregate: Dict[str, float]):
    names, values = [], []
    for m in _METRIC_ORDER:
        v = aggregate.get(m)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            names.append(_pretty(m))
            values.append(round(float(v), 3))
    return names, values


def metrics_radar(
    aggregate: Dict[str, float], config: Optional[AppConfig] = None
) -> go.Figure:
    """Radar/spider chart of aggregate metrics (all in [0, 1])."""
    names, values = _present_metrics(aggregate)
    if names:
        # Close the polygon.
        names_closed = names + [names[0]]
        values_closed = values + [values[0]]
    else:
        names_closed, values_closed = [], []
    fig = go.Figure(
        go.Scatterpolar(r=values_closed, theta=names_closed, fill="toself",
                        line=dict(color=PALETTE[0]))
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    return apply_layout(fig, "RAG quality profile (RAGAS metrics)", config, showlegend=False)


def metrics_bar(
    aggregate: Dict[str, float], config: Optional[AppConfig] = None
) -> go.Figure:
    """Bar chart of aggregate metrics with value labels."""
    names, values = _present_metrics(aggregate)
    fig = go.Figure(
        go.Bar(x=names, y=values, text=[f"{v:.2f}" for v in values], textposition="auto",
               marker=dict(color=values, colorscale="RdYlGn", cmin=0, cmax=1))
    )
    fig.update_yaxes(title="Score", range=[0, 1])
    return apply_layout(fig, "Evaluation metrics", config, showlegend=False)


def per_sample_heatmap(
    per_sample: List[Dict[str, float]], config: Optional[AppConfig] = None
) -> go.Figure:
    """Heatmap of metric scores per evaluated question."""
    if not per_sample:
        return apply_layout(go.Figure(), "Per-sample metrics", config)
    metrics = [m for m in _METRIC_ORDER if m in per_sample[0]]
    z = [[row.get(m, float("nan")) for m in metrics] for row in per_sample]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[_pretty(m) for m in metrics],
            y=[f"Q{i + 1}" for i in range(len(per_sample))],
            colorscale="RdYlGn", zmin=0, zmax=1,
            colorbar=dict(title="Score"),
        )
    )
    return apply_layout(fig, "Per-sample metric scores", config)
