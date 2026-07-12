"""Pure Plotly figure builders, free of any Streamlit dependency."""

from src.utils import env as _env  # noqa: F401

from src.visualization.corpus_viz import (
    chunk_length_distribution,
    chunks_per_document,
    document_length_distribution,
    document_summary_table,
    document_type_breakdown,
)
from src.visualization.embedding_viz import embedding_scatter
from src.visualization.evaluation_viz import (
    metrics_bar,
    metrics_radar,
    per_sample_heatmap,
)
from src.visualization.performance_viz import (
    embedding_model_comparison,
    latency_breakdown,
    stage_timing_bar,
    token_usage_bar,
    vector_store_comparison,
)
from src.visualization.retrieval_viz import (
    retrieval_scores_bar,
    score_decay_by_rank,
    score_distribution,
)

__all__ = [
    "document_summary_table",
    "document_length_distribution",
    "chunk_length_distribution",
    "chunks_per_document",
    "document_type_breakdown",
    "embedding_scatter",
    "retrieval_scores_bar",
    "score_decay_by_rank",
    "score_distribution",
    "stage_timing_bar",
    "latency_breakdown",
    "token_usage_bar",
    "embedding_model_comparison",
    "vector_store_comparison",
    "metrics_radar",
    "metrics_bar",
    "per_sample_heatmap",
]
