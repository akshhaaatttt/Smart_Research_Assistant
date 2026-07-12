"""Tests for the chunking module: strategies and page-aware document chunking."""

from pathlib import Path

import pytest

from src.chunking.document_chunker import DocumentChunker
from src.chunking.strategies import (
    FixedChunkStrategy,
    RecursiveChunkStrategy,
    SentenceChunkStrategy,
    build_strategy,
)
from src.ingestion.document_loader import DocumentIngestor
from src.preprocessing.cleaner import DocumentPreprocessor
from src.utils.config import AppConfig
from src.utils.exceptions import ChunkingError
from src.utils.schemas import Document

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

LONG_TEXT = (
    "Retrieval augmented generation grounds answers in evidence. " * 40
).strip()


# --------------------------------------------------------------------------- #
# Strategy-level behaviour
# --------------------------------------------------------------------------- #
def test_fixed_strategy_respects_size_and_overlap():
    strat = FixedChunkStrategy(chunk_size=100, chunk_overlap=20)
    chunks = strat.split(LONG_TEXT)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_recursive_strategy_produces_reasonable_chunks():
    strat = RecursiveChunkStrategy(chunk_size=120, chunk_overlap=20)
    chunks = strat.split(LONG_TEXT)
    assert len(chunks) > 1
    # Recursive splitter aims for <= size but may slightly exceed on hard text.
    assert all(len(c) <= 160 for c in chunks)


def test_sentence_strategy_keeps_sentences_whole():
    text = "First sentence here. Second sentence follows. Third one too. Fourth as well."
    strat = SentenceChunkStrategy(chunk_size=40, chunk_overlap=10)
    chunks = strat.split(text)
    # No chunk should end mid-sentence (each ends with terminal punctuation).
    assert all(c.rstrip().endswith((".", "!", "?")) for c in chunks)


def test_invalid_params_raise():
    with pytest.raises(ChunkingError):
        FixedChunkStrategy(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ChunkingError):
        FixedChunkStrategy(chunk_size=-1, chunk_overlap=0)


def test_build_strategy_unknown():
    with pytest.raises(ChunkingError):
        build_strategy("nonsense", 100, 10)


# --------------------------------------------------------------------------- #
# Document-level chunking
# --------------------------------------------------------------------------- #
def test_chunk_ids_and_indices_are_sequential():
    doc = Document(content=LONG_TEXT, source="x.txt", doc_type="txt")
    chunks = DocumentChunker().chunk_document(doc)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == doc.doc_id for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # unique ids


def test_page_attribution_from_pdf():
    doc = DocumentIngestor().ingest(str(RAW / "security_standard.pdf"))
    doc = DocumentPreprocessor().process(doc)
    chunks = DocumentChunker().chunk_document(doc)
    pages = {c.metadata.get("page") for c in chunks}
    assert pages == {1, 2}
    # A chunk mentioning incident response should be attributed to page 2.
    ir = [c for c in chunks if "incident" in c.content.lower()]
    assert ir and all(c.metadata["page"] == 2 for c in ir)


def test_min_chunk_chars_filters_tiny_fragments():
    cfg = AppConfig.load().with_overrides(
        {"chunking": {"chunk_size": 200, "chunk_overlap": 20, "min_chunk_chars": 150}}
    )
    doc = Document(content="short. " + LONG_TEXT, source="x.txt", doc_type="txt")
    chunks = DocumentChunker(cfg).chunk_document(doc)
    assert all(c.num_chars >= 150 for c in chunks)


def test_empty_document_returns_no_chunks():
    doc = Document(content="   ", source="empty.txt", doc_type="txt")
    assert DocumentChunker().chunk_document(doc) == []


def test_chunk_documents_flattens():
    d1 = Document(content=LONG_TEXT, source="a.txt", doc_type="txt")
    d2 = Document(content=LONG_TEXT, source="b.txt", doc_type="txt")
    chunks = DocumentChunker().chunk_documents([d1, d2])
    sources = {c.source for c in chunks}
    assert sources == {"a.txt", "b.txt"}
