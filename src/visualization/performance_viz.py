"""Performance and comparison visualizations: timings, latency, token usage, and model/backend comparisons."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import AppConfig
from src.visualization.theme import apply_layout


def stage_timing_bar(
    per_stage_seconds: Dict[str, float], config: Optional[AppConfig] = None
) -> go.Figure:
    """Bar chart of indexing time spent per pipeline stage."""
    stages = list(per_stage_seconds.keys())
    times = [per_stage_seconds[s] for s in stages]
    fig = go.Figure(go.Bar(x=stages, y=times, text=[f"{t:.2f}s" for t in times],
                           textposition="auto"))
    fig.update_xaxes(title="Pipeline stage")
    fig.update_yaxes(title="Seconds")
    return apply_layout(fig, "Indexing time per stage", config, showlegend=False)


def latency_breakdown(
    timings: Dict[str, float], config: Optional[AppConfig] = None
) -> go.Figure:
    """Stacked bar of query latency split into retrieval vs. generation."""
    fig = go.Figure()
    for stage, seconds in timings.items():
        fig.add_trace(go.Bar(name=stage.capitalize(), x=["query"], y=[seconds]))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Seconds")
    return apply_layout(fig, "Query latency breakdown", config, legend_title="stage")


def token_usage_bar(
    usage: Dict[str, int], config: Optional[AppConfig] = None
) -> go.Figure:
    """Bar chart of prompt vs. completion token usage."""
    keys = [k for k in ("prompt_tokens", "completion_tokens") if k in usage]
    values = [usage[k] for k in keys]
    labels = [k.replace("_", " ").title() for k in keys]
    fig = go.Figure(go.Bar(x=labels, y=values, text=values, textposition="auto"))
    fig.update_yaxes(title="Tokens")
    return apply_layout(fig, "Token usage", config, showlegend=False)


def embedding_model_comparison(
    rows: List[Dict], config: Optional[AppConfig] = None
) -> go.Figure:
    """Grouped bars comparing embedding models on speed and retrieval quality."""
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Encode time (s)", x=df["model"], y=df["encode_time_s"],
                         yaxis="y1"))
    fig.add_trace(go.Scatter(name="Avg top-1 similarity", x=df["model"],
                             y=df["avg_top1_similarity"], yaxis="y2",
                             mode="lines+markers"))
    fig.update_layout(
        yaxis=dict(title="Encode time (s)"),
        yaxis2=dict(title="Avg top-1 similarity", overlaying="y", side="right",
                    range=[0, 1]),
    )
    return apply_layout(fig, "Embedding model comparison", config, legend_title="metric")


def vector_store_comparison(
    rows: List[Dict], config: Optional[AppConfig] = None
) -> go.Figure:
    """Grouped bars comparing vector-store backends on index/query time and storage."""
    df = pd.DataFrame(rows)
    metrics = [("index_time_s", "Index time (s)"),
               ("query_time_ms", "Query time (ms)"),
               ("storage_mb", "Storage (MB)")]
    fig = go.Figure()
    for col, label in metrics:
        if col in df.columns:
            fig.add_trace(go.Bar(name=label, x=df["backend"], y=df[col]))
    fig.update_layout(barmode="group")
    return apply_layout(fig, "Vector store comparison", config, legend_title="metric")
