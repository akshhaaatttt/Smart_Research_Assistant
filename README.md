# 📚 Smart Research Assistant — RAG-Based Knowledge System

An end-to-end **Retrieval-Augmented Generation (RAG)** assistant that lets you upload
documents (PDF, TXT, DOCX, Markdown, or web pages), ask questions in natural language,
and get **accurate, context-grounded answers with source attribution and quality
metrics**.

The project implements the complete lifecycle of a modern RAG pipeline as independent,
tested modules, integrated behind a single pipeline class and an interactive Streamlit
application.

> **Runs fully offline out of the box.** Embeddings use local Sentence-Transformers and
> generation falls back to a local HuggingFace model, so **no API key is required**.
> Configure an OpenAI-compatible endpoint to upgrade answer quality and enable
> LLM-judged RAGAS evaluation.

---

## ✨ Features

| Area | What it does |
|------|--------------|
| **Ingestion** | PDF, TXT, DOCX, Markdown, and web URLs → uniform `Document` objects with metadata (pages, title, author, …) |
| **Preprocessing** | Unicode normalization, PDF de-hyphenation, boilerplate/page-number stripping, whitespace cleanup — page boundaries preserved for citation |
| **Chunking** | Recursive / fixed / sentence strategies; configurable size & overlap; **per-page attribution** |
| **Embeddings** | Multiple Sentence-Transformer models (MiniLM, MPNet, BGE), swappable and cached |
| **Vector store** | **FAISS** *and* **ChromaDB** behind one interface; persist/reload; identical score contract |
| **Retrieval** | Top-K search, absolute + **query-adaptive relative** score thresholds, optional **MMR** re-ranking |
| **Generation** | Pluggable LLM: OpenAI-compatible endpoint **or** local FLAN-T5; strictly grounded prompt to curb hallucination |
| **Evaluation** | **RAGAS** (LLM-judged) when a key is present, else offline embedding-based metric approximations |
| **Visualization** | ~17 Plotly charts: corpus stats, PCA/t-SNE embedding maps, retrieval scores, latency/tokens, model & DB comparison, RAGAS dashboard |
| **App** | Streamlit UI with upload, chat, retrieved-context viewer, and evaluation / performance / comparison dashboards |

---

## 🏗️ Architecture

```
User upload / URL
      │
      ▼
┌───────────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────────┐
│  Ingestion    │──▶│ Preprocessing│──▶│ Chunking │──▶│  Embeddings │
└───────────────┘   └──────────────┘   └──────────┘   └──────┬──────┘
                                                             ▼
                                                    ┌──────────────────┐
                                                    │  Vector store    │
                                                    │  FAISS / Chroma  │
                                                    └────────┬─────────┘
                                                             ▼
   Question ──▶ embed ──▶ ┌───────────┐ ──▶ ┌───────────────┐ ──▶ Answer
                          │ Retriever │     │  Prompt + LLM │      + sources
                          └───────────┘     └───────────────┘      + timings
                                                             │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │   Evaluation     │
                                                    │  RAGAS / builtin │
                                                    └──────────────────┘
```

Every stage is a standalone, unit-tested module; `src/pipeline.py` wires them into the
indexing and querying flows.

---

## 📁 Project structure

```
RAG-System/
├── app/                    # Streamlit application
│   ├── streamlit_app.py    #   tabbed UI (Documents / Chat / Evaluation / Performance / Comparison)
│   └── state.py            #   session-state & pipeline lifecycle helpers
├── config/
│   └── config.yaml         # single source of truth for every tunable parameter
├── data/
│   ├── raw/                # sample documents (HR policy, security PDF, benefits DOCX, RAG overview)
│   ├── processed/ chunks/  # generated intermediates
│   └── uploads/            # files uploaded through the UI
├── notebooks/
│   └── rag_walkthrough.ipynb  # step-by-step tour of the pipeline
├── src/
│   ├── ingestion/          # loaders (PDF/TXT/DOCX/MD/web) + dispatcher
│   ├── preprocessing/      # text cleaning
│   ├── chunking/           # splitting strategies + page-aware chunker
│   ├── embeddings/         # embedding models + manager
│   ├── retrieval/          # FAISS/Chroma stores, factory, retriever
│   ├── generation/         # prompt builder, LLM backends, answer generator
│   ├── evaluation/         # metrics, RAGAS adapter, unified evaluator, benchmarks
│   ├── visualization/      # Plotly chart builders
│   ├── utils/              # config, logging, exceptions, schemas, env bootstrap
│   └── pipeline.py         # end-to-end RAG orchestrator
├── tests/                  # pytest suite (99+ tests across all modules)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. (Optional) configure a hosted LLM

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY (and OPENAI_BASE_URL for Groq/Azure/Ollama/vLLM/etc.)
```

For example, to use **Groq** (fast, free-tier hosted Llama models):

```ini
OPENAI_API_KEY=gsk_your_groq_key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

With a hosted model configured, generation quality jumps (grounded, cited answers on
dense technical PDFs) and **RAGAS** evaluation activates automatically — using the
hosted LLM as judge and the local embeddings for similarity, so it works even with
providers (like Groq) that have no embeddings endpoint.

Without this, the assistant uses a **local** model — the first run downloads the
embedding model (~90 MB) and the local LLM (FLAN-T5, ~1 GB).

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501>, go to **Documents → Load sample documents** (or upload
your own), then ask questions on the **Chat** tab.

### 4. Or use the pipeline programmatically

```python
from src.pipeline import RAGPipeline

pipe = RAGPipeline()
pipe.index_directory("data/raw")

resp = pipe.query("What is the maternity leave policy?")
print(resp.answer)        # -> "Eligible employees are entitled to 26 weeks ..."
print(resp.sources)       # -> ["hr_policy.txt, page 1", ...]
print(resp.timings)       # -> {"retrieval": 0.01, "generation": 3.2}
```

---

## ⚙️ Configuration

All behaviour is controlled from [`config/config.yaml`](config/config.yaml) — nothing is
hardcoded. Key knobs:

- `embeddings.active` — which embedding model to use (`minilm`, `mpnet`, `bge-small`)
- `vector_store.active` — `faiss` or `chroma`
- `chunking.{strategy,chunk_size,chunk_overlap}`
- `retrieval.{top_k,score_threshold,relative_score_floor,mmr}`
- `generation.{backend,temperature,max_tokens}` (`backend: auto` picks OpenAI if a key
  is set, else the local model)
- `evaluation.use_ragas`

Values can be overridden at runtime — the Streamlit sidebar does exactly this via
`AppConfig.with_overrides(...)`.

---

## 🧪 Testing

```bash
pytest                     # full suite (loads models; a few minutes on first run)
pytest -m "not integration"  # fast unit tests only
```

The suite covers every module: ingestion across all formats, cleaning, chunking
strategies & page attribution, embedding semantics, both vector stores (parametrized),
retrieval + MMR + thresholds, prompt/generation/backends, evaluation metrics,
visualizations, benchmarks, the end-to-end pipeline, and the Streamlit app (via
`AppTest`).

---

## 📊 Evaluation metrics

| Metric | Question it answers |
|--------|--------------------|
| **Faithfulness** | Is the answer supported by the retrieved context? (anti-hallucination) |
| **Answer relevancy** | Does the answer address the question? |
| **Context precision** | Are the retrieved chunks on-topic? |
| **Context recall** | Does the context cover the reference answer? *(needs ground truth)* |
| **Answer correctness** | Does the answer match the reference? *(needs ground truth)* |

With an API key, these are computed by **RAGAS** (LLM-judged). Offline, they use
**embedding-based approximations** that correlate with the LLM-judged versions and are
fast, free, and reproducible — ideal for model/backend sweeps.

---

## 🛠️ Design notes

- **Offline-first, pluggable LLM.** The `generation` layer is decoupled from any
  provider via a `BaseLLM` interface, so the same pipeline runs against a local model
  or any OpenAI-compatible endpoint.
- **Anti-hallucination.** The prompt instructs the model to answer *only* from context
  and to explicitly decline when the answer is absent; a query-adaptive relevance floor
  keeps weak passages out of the LLM's context.
- **Consistent score contract.** FAISS (inner product on normalized vectors) and Chroma
  (cosine distance) both return cosine similarity in `[-1, 1]`, so downstream code is
  backend-agnostic.
- **Environment robustness.** `src/utils/env.py` forces HuggingFace onto the PyTorch
  backend (the local Anaconda env ships TensorFlow/Keras 3, which otherwise breaks
  `transformers` on import) and loads `.env`.

---

## 🔮 Future enhancements

Multi-turn conversational memory · voice interaction · multi-language support ·
cloud deployment · authentication · domain-specific assistants.
