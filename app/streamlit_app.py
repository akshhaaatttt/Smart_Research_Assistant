"""Smart Research Assistant - Streamlit application.

Run with ``streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable: `streamlit run` puts app/ on sys.path, not the root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app import state, styles  # noqa: E402
from app.state import (
    default_settings,
    ensure_session_defaults,
    get_pipeline,
    save_uploads,
)

import src.visualization as viz  # noqa: E402
from src.evaluation.benchmark import (  # noqa: E402
    compare_embedding_models,
    compare_vector_stores,
)
from src.evaluation.evaluator import RAGEvaluator  # noqa: E402
from src.utils.exceptions import RAGError  # noqa: E402

st.set_page_config(page_title="Smart Research Assistant", page_icon="📚", layout="wide")

RAW_DIR = state._ROOT / "data" / "raw"


def _llm_label(pipe) -> str:
    """Friendly provider label, e.g. 'Groq · llama-4-scout' or 'Local · flan-t5-base'."""
    llm = pipe.generator.llm
    model = (llm.model or "").split("/")[-1]
    if llm.backend == "openai":
        base = (getattr(llm, "base_url", "") or "").lower()
        provider = "Groq" if "groq" in base else ("OpenAI" if "openai" in base else "API")
    else:
        provider = "Local"
    return f"{provider} · {model}"

# Curated demo questions (with ground truth) for the evaluation tab.
DEMO_EVAL = [
    ("What is the maternity leave policy?",
     "Employees are entitled to 26 weeks of paid maternity leave after 12 months of service."),
    ("How many days of annual leave do employees get?",
     "Full-time employees get 24 days of paid annual leave per year."),
    ("How must customer data be encrypted at rest?",
     "Customer data must be encrypted at rest using AES-256."),
    ("How much does the company match on 401(k) contributions?",
     "The company matches 401(k) contributions up to five percent of base salary."),
]


def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Configuration")
    defaults = default_settings()
    cfg = state.AppConfig.load()

    st.sidebar.subheader("Models & storage")
    embedding_key = st.sidebar.selectbox(
        "Embedding model", options=list(cfg.get("embeddings.models").keys()),
        index=list(cfg.get("embeddings.models").keys()).index(defaults["embedding_key"]),
        help="Different models trade off speed vs. retrieval quality. Changing this "
             "rebuilds the index.",
    )
    backend = st.sidebar.selectbox(
        "Vector database", options=["faiss", "chroma"],
        index=["faiss", "chroma"].index(defaults["backend"]),
        help="FAISS is fast and in-memory; ChromaDB is persistent. Changing this "
             "rebuilds the index.",
    )

    st.sidebar.subheader("Chunking")
    chunk_size = st.sidebar.slider("Chunk size (chars)", 200, 2000, defaults["chunk_size"], 50)
    chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", 0, 500, defaults["chunk_overlap"], 10)

    st.sidebar.subheader("Retrieval & generation")
    top_k = st.sidebar.slider("Top-K retrieved", 1, 10, defaults["top_k"])
    temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, float(defaults["temperature"]), 0.05)

    settings = {
        "embedding_key": embedding_key, "backend": backend,
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "top_k": top_k, "temperature": temperature,
    }

    st.sidebar.divider()
    if st.sidebar.button("🔄 Reset index", use_container_width=True):
        pipe = get_pipeline(settings)
        pipe.reset_index()
        st.session_state.index_stats = None
        st.session_state.chat_history = []
        st.sidebar.success("Index cleared.")

    pipe = get_pipeline(settings)
    st.sidebar.divider()
    st.sidebar.markdown("**System status** <span class='badge'>live</span>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<div class='side-card'>"
        f"📦 <b>Indexed chunks:</b> {pipe.num_indexed}<br>"
        f"🧠 <b>Embedding:</b> {pipe.embedding_key} ({pipe.embedding_model.dimension}d)<br>"
        f"🗄️ <b>Vector DB:</b> {pipe.backend}<br>"
        f"🤖 <b>LLM:</b> {_llm_label(pipe)}"
        f"</div>",
        unsafe_allow_html=True,
    )
    return settings


def tab_documents(settings: dict) -> None:
    pipe = get_pipeline(settings)
    st.header("📥 Documents")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Upload files** (PDF, TXT, DOCX, MD)")
        uploaded = st.file_uploader(
            "Drop files here", type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True, label_visibility="collapsed",
        )
    with col2:
        st.markdown("**Add web pages** (one URL per line)")
        urls_text = st.text_area("URLs", height=110, label_visibility="collapsed",
                                 placeholder="https://example.com/article")

    c1, c2, c3 = st.columns(3)
    build = c1.button("🚀 Build / update index", type="primary", use_container_width=True)
    load_samples = c2.button("📄 Load sample documents", use_container_width=True)
    show_stats = c3.button("📊 Refresh corpus stats", use_container_width=True)

    sources: list[str] = []
    if build:
        if uploaded:
            sources += save_uploads(uploaded)
        sources += [u.strip() for u in urls_text.splitlines() if u.strip()]
    if load_samples:
        sources += [str(p) for p in RAW_DIR.glob("*") if p.is_file()]

    if sources:
        with st.spinner(f"Indexing {len(sources)} source(s)..."):
            stats = pipe.index_sources(sources)
            st.session_state.index_stats = stats
        st.success(
            f"Indexed **{stats.documents}** document(s) into **{stats.chunks}** chunks "
            f"in {stats.elapsed_seconds:.2f}s."
        )

    if pipe.documents:
        st.subheader("Corpus overview")
        st.dataframe(viz.document_summary_table(pipe.documents), use_container_width=True,
                     hide_index=True)
        g1, g2 = st.columns(2)
        g1.plotly_chart(viz.document_type_breakdown(pipe.documents), use_container_width=True)
        g2.plotly_chart(viz.chunks_per_document(pipe.chunks), use_container_width=True)
        g3, g4 = st.columns(2)
        g3.plotly_chart(viz.document_length_distribution(pipe.documents), use_container_width=True)
        g4.plotly_chart(viz.chunk_length_distribution(pipe.chunks), use_container_width=True)
    elif pipe.num_indexed:
        st.info(f"{pipe.num_indexed} chunks are indexed from a previous session. "
                "Load or upload documents to see corpus statistics.")
    else:
        st.info("No documents indexed yet. Upload files, add URLs, or load the samples.")


def _render_sources(response) -> None:
    st.markdown("**📎 Sources**")
    for s in response.sources:
        st.markdown(f"- {s}")
    with st.expander(f"🔍 Retrieved context ({len(response.contexts)} passages)"):
        for rc in response.contexts:
            meta = rc.chunk.metadata
            label = meta.get("filename") or meta.get("title") or rc.source
            page = f" · page {meta['page']}" if "page" in meta else ""
            styles.source_card(f"[{rc.rank + 1}] {label}{page}", rc.score, rc.content)


def _conversation_history() -> list:
    """Build (question, answer) turns from the chat log for multi-turn memory."""
    turns, pending_q = [], None
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            pending_q = turn["content"]
        elif turn["role"] == "assistant" and pending_q is not None:
            turns.append((pending_q, turn["content"]))
            pending_q = None
    return turns


def tab_chat(settings: dict) -> None:
    pipe = get_pipeline(settings)
    head, clear = st.columns([5, 1])
    head.header("💬 Ask a question")
    if clear.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.rerun()

    if pipe.num_indexed == 0:
        st.warning("Index is empty. Add documents on the **Documents** tab first.")
        return

    st.caption("💡 Multi-turn memory is on — you can ask follow-up questions "
               "(e.g. \"and what about paternity leave?\").")

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("response") is not None:
                _render_sources(turn["response"])

    question = st.chat_input("e.g. What is the maternity leave policy?")
    if question:
        history = _conversation_history()  # prior turns, before this question
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                response = pipe.query(question, top_k=settings["top_k"], history=history)
            st.markdown(response.answer or "_No answer generated._")
            _render_sources(response)
            usage = response.metadata.get("usage", {})
            st.caption(
                f"⏱ retrieval {response.timings.get('retrieval', 0):.2f}s · "
                f"generation {response.timings.get('generation', 0):.2f}s · "
                f"{usage.get('total_tokens', 0)} tokens · "
                f"{response.metadata.get('llm_backend', '?')}"
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response.answer, "response": response}
        )
        st.session_state.last_response = response

    # Retrieval diagnostics for the most recent answer.
    if st.session_state.last_response and st.session_state.last_response.contexts:
        st.divider()
        st.subheader("Retrieval diagnostics (last question)")
        d1, d2 = st.columns(2)
        d1.plotly_chart(viz.retrieval_scores_bar(st.session_state.last_response.contexts),
                        use_container_width=True)
        d2.plotly_chart(viz.score_decay_by_rank(st.session_state.last_response.contexts),
                        use_container_width=True)


def tab_evaluation(settings: dict) -> None:
    pipe = get_pipeline(settings)
    st.header("📊 RAG Evaluation")
    st.caption(
        "Metrics are computed with RAGAS when an OpenAI-compatible key is configured, "
        "otherwise with built-in embedding-based approximations (offline)."
    )

    if pipe.num_indexed == 0:
        st.warning("Index is empty. Add documents first.")
        return

    evaluator = RAGEvaluator(pipe.embedding_model, pipe.config)
    st.info(f"Evaluation method in use: **{evaluator.method_in_use()}**")

    use_demo = st.checkbox("Use demo question set (HR / security / benefits)", value=True)
    if use_demo:
        questions = [q for q, _ in DEMO_EVAL]
        truths = [gt for _, gt in DEMO_EVAL]
        st.dataframe({"question": questions, "ground_truth": truths},
                     use_container_width=True, hide_index=True)
    else:
        raw = st.text_area(
            "Questions (one per line, optionally `question | ground truth`)",
            height=150,
        )
        pairs = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            if "|" in line:
                q, gt = line.split("|", 1)
                pairs.append((q.strip(), gt.strip()))
            else:
                pairs.append((line.strip(), None))
        questions = [q for q, _ in pairs]
        truths = [gt for _, gt in pairs]

    if st.button("▶ Run evaluation", type="primary") and questions:
        with st.spinner(f"Evaluating {len(questions)} question(s)..."):
            report = evaluator.evaluate_pipeline(pipe, questions, truths, top_k=settings["top_k"])
            st.session_state.eval_report = report

    report = st.session_state.eval_report
    if report:
        st.subheader(f"Results ({report.method})")
        cols = st.columns(len(report.aggregate) or 1)
        for col, (metric, value) in zip(cols, report.aggregate.items()):
            display = "n/a" if value != value else f"{value:.3f}"  # NaN check
            col.metric(metric.replace("_", " ").title(), display)

        v1, v2 = st.columns(2)
        v1.plotly_chart(viz.metrics_radar(report.aggregate), use_container_width=True)
        v2.plotly_chart(viz.metrics_bar(report.aggregate), use_container_width=True)
        if report.per_sample:
            st.plotly_chart(viz.per_sample_heatmap(report.per_sample), use_container_width=True)


def tab_performance(settings: dict) -> None:
    pipe = get_pipeline(settings)
    st.header("⚡ Performance & Embedding Space")

    stats = st.session_state.index_stats
    if stats:
        st.subheader("Indexing performance")
        p1, p2 = st.columns(2)
        p1.plotly_chart(viz.stage_timing_bar(stats.per_stage_seconds), use_container_width=True)
        p2.metric("Total indexing time", f"{stats.elapsed_seconds:.2f}s")
        p2.metric("Chunks / second",
                  f"{stats.chunks / stats.elapsed_seconds:.1f}" if stats.elapsed_seconds else "—")

    resp = st.session_state.last_response
    if resp:
        st.subheader("Last query performance")
        q1, q2 = st.columns(2)
        q1.plotly_chart(viz.latency_breakdown(resp.timings), use_container_width=True)
        usage = resp.metadata.get("usage", {})
        if usage:
            q2.plotly_chart(viz.token_usage_bar(usage), use_container_width=True)

    if pipe.chunks:
        st.subheader("Embedding space projection")
        method = st.radio("Projection method", ["pca", "tsne"], horizontal=True)
        with st.spinner("Projecting embeddings..."):
            texts = [c.content for c in pipe.chunks]
            labels = [c.metadata.get("filename", c.source) for c in pipe.chunks]
            embeddings = pipe.embedding_model.embed_texts(texts)
            query_vec = None
            if resp is not None:
                query_vec = pipe.embedding_model.embed_query(resp.query)
            fig = viz.embedding_scatter(
                embeddings, labels, method=method,
                hover_texts=[t[:80] for t in texts],
                query_embedding=query_vec, config=pipe.config,
            )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each point is a chunk; the ★ (if present) is your last question. "
                   "Nearby points are semantically similar.")
    else:
        st.info("Index some documents to visualize the embedding space and performance.")


def tab_comparison(settings: dict) -> None:
    pipe = get_pipeline(settings)
    st.header("🔬 Model & Backend Comparison")
    st.caption("Benchmark embedding models and vector databases on your indexed corpus.")

    if not pipe.chunks:
        st.warning("Index some documents first to run comparisons.")
        return

    chunk_texts = [c.content for c in pipe.chunks]
    probe_queries = [
        "What is the maternity leave policy?",
        "How is customer data protected?",
        "What retirement benefits are offered?",
    ]

    st.subheader("Embedding models")
    all_models = list(pipe.config.get("embeddings.models").keys())
    chosen = st.multiselect(
        "Models to compare (unselected models download on first use)",
        options=all_models, default=[pipe.embedding_key],
    )
    if st.button("Compare embedding models") and chosen:
        with st.spinner("Encoding corpus with each model..."):
            rows = compare_embedding_models(chunk_texts, probe_queries, chosen, pipe.config)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.plotly_chart(viz.embedding_model_comparison(rows), use_container_width=True)

    st.divider()
    st.subheader("Vector databases (FAISS vs ChromaDB)")
    if st.button("Compare vector stores"):
        with st.spinner("Indexing and querying with each backend..."):
            embeddings = pipe.embedding_model.embed_texts(chunk_texts)
            query = pipe.embedding_model.embed_query(probe_queries[0])
            rows = compare_vector_stores(
                pipe.chunks, embeddings, query, ["faiss", "chroma"], pipe.config,
                top_k=settings["top_k"],
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.plotly_chart(viz.vector_store_comparison(rows), use_container_width=True)


def main() -> None:
    ensure_session_defaults()
    styles.inject()
    styles.hero(
        "Smart Research Assistant",
        "Retrieval-Augmented Generation over your documents — grounded answers with "
        "sources and evaluation.",
    )

    try:
        settings = render_sidebar()
    except RAGError as exc:
        st.error(f"⚠️ Could not initialize the assistant.\n\n{exc}")
        st.info(
            "This usually means the machine is low on RAM while loading the embedding "
            "model. Close other applications (browsers, IDEs) to free memory, then click "
            "**Rerun** (top-right menu) or refresh the page."
        )
        st.stop()

    pipe = get_pipeline(settings)
    styles.kpi_row([
        ("Documents", str(len(pipe.documents) or "—"), "in corpus"),
        ("Chunks indexed", str(pipe.num_indexed), "searchable"),
        ("Embedding", pipe.embedding_key, f"{pipe.embedding_model.dimension}-dim"),
        ("LLM", _llm_label(pipe).split(" · ")[0], _llm_label(pipe).split(" · ")[-1]),
    ])

    tabs = st.tabs(
        ["📥 Documents", "💬 Chat", "📊 Evaluation", "⚡ Performance", "🔬 Comparison"]
    )
    with tabs[0]:
        tab_documents(settings)
    with tabs[1]:
        tab_chat(settings)
    with tabs[2]:
        tab_evaluation(settings)
    with tabs[3]:
        tab_performance(settings)
    with tabs[4]:
        tab_comparison(settings)


if __name__ == "__main__":
    main()
