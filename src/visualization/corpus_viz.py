"""Corpus-level visualizations: document and chunk statistics."""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import AppConfig
from src.utils.schemas import Chunk, Document
from src.visualization.theme import apply_layout


def document_summary_table(documents: List[Document]) -> pd.DataFrame:
    """Tabular per-document summary (chars, words, type)."""
    rows = []
    for d in documents:
        rows.append(
            {
                "source": d.metadata.get("filename", d.source),
                "type": d.doc_type,
                "characters": d.num_chars,
                "words": d.num_words,
                # Single str type so Streamlit/Arrow serialize cleanly.
                "pages": str(d.metadata.get("num_pages", "-")),
            }
        )
    return pd.DataFrame(rows)


def document_length_distribution(
    documents: List[Document], config: Optional[AppConfig] = None
) -> go.Figure:
    """Histogram of document lengths (in words)."""
    lengths = [d.num_words for d in documents]
    fig = px.histogram(x=lengths, nbins=max(5, min(20, len(lengths))))
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    fig.update_xaxes(title="Document length (words)")
    fig.update_yaxes(title="Number of documents")
    return apply_layout(fig, "Document length distribution", config, showlegend=False)


def chunk_length_distribution(
    chunks: List[Chunk], config: Optional[AppConfig] = None
) -> go.Figure:
    """Histogram of chunk lengths (in characters)."""
    lengths = [c.num_chars for c in chunks]
    fig = px.histogram(x=lengths, nbins=30)
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    fig.update_xaxes(title="Chunk length (characters)")
    fig.update_yaxes(title="Number of chunks")
    return apply_layout(fig, "Chunk length distribution", config, showlegend=False)


def chunks_per_document(
    chunks: List[Chunk], config: Optional[AppConfig] = None
) -> go.Figure:
    """Bar chart of how many chunks each source produced."""
    counts = Counter(c.metadata.get("filename", c.source) for c in chunks)
    items = counts.most_common()
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h"))
    fig.update_xaxes(title="Number of chunks")
    fig.update_yaxes(title="Source", autorange="reversed")
    return apply_layout(fig, "Chunks per document", config, showlegend=False)


def document_type_breakdown(
    documents: List[Document], config: Optional[AppConfig] = None
) -> go.Figure:
    """Donut chart of document types in the corpus."""
    counts = Counter(d.doc_type for d in documents)
    fig = go.Figure(
        go.Pie(labels=list(counts.keys()), values=list(counts.values()), hole=0.45)
    )
    return apply_layout(fig, "Document types", config)
