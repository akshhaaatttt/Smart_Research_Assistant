"""Streamlit session/state helpers.

The pipeline is expensive to build (it loads an embedding model), so it is created once
and rebuilt only when a setting that changes the vector space is applied.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Make the project root importable when launched via `streamlit run app/...`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from src.pipeline import RAGPipeline  # noqa: E402
from src.utils.config import AppConfig  # noqa: E402

UPLOAD_DIR = _ROOT / "data" / "uploads"


def _build_overrides(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Translate UI settings into a config-override dict."""
    return {
        "chunking": {
            "chunk_size": settings["chunk_size"],
            "chunk_overlap": settings["chunk_overlap"],
        },
        "retrieval": {"top_k": settings["top_k"]},
        "generation": {"temperature": settings["temperature"]},
        "embeddings": {"active": settings["embedding_key"]},
        "vector_store": {"active": settings["backend"]},
    }


def default_settings() -> Dict[str, Any]:
    cfg = AppConfig.load()
    return {
        "embedding_key": cfg.get("embeddings.active"),
        "backend": cfg.get("vector_store.active"),
        "chunk_size": cfg.get("chunking.chunk_size"),
        "chunk_overlap": cfg.get("chunking.chunk_overlap"),
        "top_k": cfg.get("retrieval.top_k"),
        "temperature": cfg.get("generation.temperature"),
    }


def get_pipeline(settings: Dict[str, Any], *, force_rebuild: bool = False) -> RAGPipeline:
    """Return a pipeline matching ``settings``, rebuilding only when needed."""
    cfg = AppConfig.load().with_overrides(_build_overrides(settings))

    needs_rebuild = (
        force_rebuild
        or "pipeline" not in st.session_state
        or st.session_state.get("active_embedding") != settings["embedding_key"]
        or st.session_state.get("active_backend") != settings["backend"]
    )

    if needs_rebuild:
        pipe = RAGPipeline(
            cfg,
            embedding_key=settings["embedding_key"],
            backend=settings["backend"],
            auto_load=True,
        )
        st.session_state.pipeline = pipe
        st.session_state.active_embedding = settings["embedding_key"]
        st.session_state.active_backend = settings["backend"]
    else:
        # Cheap settings (top_k, temperature) are applied without a rebuild.
        pipe = st.session_state.pipeline
        pipe.config = cfg
        pipe.retriever.config = cfg
        pipe.retriever.default_top_k = settings["top_k"]
        pipe.generator = pipe.generator  # unchanged
    return st.session_state.pipeline


def ensure_session_defaults() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("index_stats", None)
    st.session_state.setdefault("eval_report", None)


def save_uploads(uploaded_files) -> list[str]:
    """Persist Streamlit UploadedFile objects to disk and return their paths."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for uf in uploaded_files:
        dest = UPLOAD_DIR / uf.name
        dest.write_bytes(uf.getbuffer())
        paths.append(str(dest))
    return paths
