"""End-to-end RAG pipeline orchestrator wiring ingestion, indexing, retrieval, and generation.

Switching the embedding model or vector-store backend changes the vector space, so those
operations rebuild the store and require re-indexing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.chunking.document_chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingManager, EmbeddingModel
from src.generation.generator import AnswerGenerator, GenerationOutput
from src.ingestion.document_loader import DocumentIngestor
from src.preprocessing.cleaner import DocumentPreprocessor
from src.retrieval.base_store import BaseVectorStore
from src.retrieval.factory import build_vector_store
from src.retrieval.retriever import Retriever
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.schemas import Chunk, Document, RAGResponse, RetrievedChunk

logger = get_logger(__name__)


@dataclass
class IndexStats:
    documents: int
    chunks: int
    embedding_model: str
    backend: str
    dimension: int
    elapsed_seconds: float
    per_stage_seconds: dict = field(default_factory=dict)


class RAGPipeline:
    """Assemble ingestion, indexing, retrieval, and generation."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        embedding_key: Optional[str] = None,
        backend: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        self.config = config or AppConfig.load()

        self.ingestor = DocumentIngestor(self.config)
        self.preprocessor = DocumentPreprocessor(self.config)
        self.chunker = DocumentChunker(self.config)
        self.embeddings = EmbeddingManager(self.config)

        self._embedding_key = embedding_key or self.embeddings.active_key
        self._backend = (backend or self.config.get("vector_store.active")).lower()

        self.embedding_model: EmbeddingModel = self.embeddings.get_model(self._embedding_key)
        self.vector_store: BaseVectorStore = build_vector_store(
            self.embedding_model.dimension, backend=self._backend, config=self.config
        )
        self.retriever = Retriever(self.embedding_model, self.vector_store, self.config)
        self.generator = AnswerGenerator(config=self.config)

        # In-memory records so viz/eval can introspect without re-reading disk.
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []

        if auto_load:
            try:
                if self.vector_store.load():
                    logger.info("Loaded existing index with %d vectors", self.vector_store.count())
            except Exception as exc:  # a stale index shouldn't crash startup
                logger.warning("Could not load existing index: %s", exc)

    @property
    def embedding_key(self) -> str:
        return self._embedding_key

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def num_indexed(self) -> int:
        return self.vector_store.count()

    def index_sources(self, sources: List[str], *, persist: bool = True) -> IndexStats:
        """Ingest, clean, chunk, embed and store the given file paths / URLs."""
        timings: dict = {}
        start = time.perf_counter()

        t = time.perf_counter()
        documents = self.ingestor.ingest_many(sources, skip_errors=True)
        timings["ingest"] = time.perf_counter() - t
        return self._index_documents(documents, timings, start, persist)

    def index_directory(self, directory: str, *, persist: bool = True) -> IndexStats:
        timings: dict = {}
        start = time.perf_counter()
        t = time.perf_counter()
        documents = self.ingestor.ingest_directory(directory, skip_errors=True)
        timings["ingest"] = time.perf_counter() - t
        return self._index_documents(documents, timings, start, persist)

    def index_documents(self, documents: List[Document], *, persist: bool = True) -> IndexStats:
        """Index already-loaded Document objects (e.g. from an upload buffer)."""
        return self._index_documents(documents, {"ingest": 0.0}, time.perf_counter(), persist)

    def _index_documents(
        self, documents: List[Document], timings: dict, start: float, persist: bool
    ) -> IndexStats:
        t = time.perf_counter()
        cleaned = self.preprocessor.process_many(documents)
        timings["clean"] = time.perf_counter() - t

        t = time.perf_counter()
        chunks = self.chunker.chunk_documents(cleaned)
        timings["chunk"] = time.perf_counter() - t

        t = time.perf_counter()
        if chunks:
            vectors = self.embedding_model.embed_texts([c.content for c in chunks])
            timings["embed"] = time.perf_counter() - t

            t = time.perf_counter()
            self.vector_store.add(chunks, vectors)
            if persist:
                self.vector_store.persist()
            timings["store"] = time.perf_counter() - t
        else:
            timings["embed"] = 0.0
            timings["store"] = 0.0

        self.documents.extend(cleaned)
        self.chunks.extend(chunks)

        elapsed = time.perf_counter() - start
        stats = IndexStats(
            documents=len(cleaned),
            chunks=len(chunks),
            embedding_model=self._embedding_key,
            backend=self._backend,
            dimension=self.embedding_model.dimension,
            elapsed_seconds=elapsed,
            per_stage_seconds=timings,
        )
        logger.info(
            "Indexed %d docs -> %d chunks in %.2fs %s",
            stats.documents, stats.chunks, elapsed,
            {k: round(v, 3) for k, v in timings.items()},
        )
        return stats

    def retrieve(self, question: str, top_k: Optional[int] = None, **kw) -> List[RetrievedChunk]:
        return self.retriever.retrieve(question, top_k=top_k, **kw)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        *,
        generate: bool = True,
        history: Optional[List[tuple]] = None,
        **retrieval_kwargs,
    ) -> RAGResponse:
        """Answer ``question`` end-to-end, returning answer + evidence + timings.

        ``history`` is an optional list of prior (question, answer) turns; when given,
        the retrieval query is augmented with the recent question(s) so follow-ups
        ("and paternity leave?") pull the right passages, and the turns are passed to
        the generator so the LLM can resolve references.
        """
        timings: dict = {}

        retrieval_query = self._augment_query(question, history)
        t = time.perf_counter()
        contexts = self.retriever.retrieve(retrieval_query, top_k=top_k, **retrieval_kwargs)
        timings["retrieval"] = time.perf_counter() - t

        answer = ""
        gen_out: Optional[GenerationOutput] = None
        if generate:
            gen_out = self.generator.generate(question, contexts, history=history)
            answer = gen_out.answer
            timings["generation"] = gen_out.latency_seconds
            # Prefer the contexts the prompt actually used (post-budget) for attribution.
            contexts = gen_out.used_contexts or contexts

        metadata = {
            "embedding_model": self._embedding_key,
            "backend": self._backend,
            "top_k": top_k if top_k is not None else self.retriever.default_top_k,
        }
        if gen_out is not None:
            metadata.update(
                {"llm_backend": gen_out.backend, "llm_model": gen_out.model, "usage": gen_out.usage}
            )

        return RAGResponse(
            query=question, answer=answer, contexts=contexts, timings=timings, metadata=metadata
        )

    @staticmethod
    def _augment_query(question: str, history: Optional[List[tuple]]) -> str:
        """Prepend the previous user question(s) so a short follow-up still retrieves
        the right chunks. Cheap and deterministic - no extra LLM call."""
        if not history:
            return question
        prev_questions = " ".join(q for q, _ in list(history)[-2:] if q)
        return f"{prev_questions} {question}".strip() if prev_questions else question

    def switch_embedding_model(self, key: str) -> None:
        """Switch embedding model. Rebuilds the store; re-indexing is required."""
        self.embedding_model = self.embeddings.set_active(key)
        self._embedding_key = key
        self._rebuild_store(reset=True)
        logger.info("Switched embedding model to '%s' (index cleared - re-index needed)", key)

    def switch_backend(self, backend: str) -> None:
        """Switch vector-store backend. Rebuilds the store; re-indexing required."""
        self._backend = backend.lower()
        self._rebuild_store(reset=True)
        logger.info("Switched vector store backend to '%s' (re-index needed)", backend)

    def _rebuild_store(self, *, reset: bool) -> None:
        self.vector_store = build_vector_store(
            self.embedding_model.dimension, backend=self._backend, config=self.config
        )
        if reset:
            self.vector_store.reset()
            self.documents.clear()
            self.chunks.clear()
        self.retriever = Retriever(self.embedding_model, self.vector_store, self.config)

    def reset_index(self) -> None:
        self.vector_store.reset()
        self.documents.clear()
        self.chunks.clear()
        logger.info("Index reset")
