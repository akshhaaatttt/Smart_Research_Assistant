"""Tests for the ingestion module across all supported formats."""

from pathlib import Path

import pytest

from src.ingestion.document_loader import DocumentIngestor
from src.ingestion.loaders import TextLoader
from src.utils.exceptions import IngestionError, UnsupportedFormatError

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def ingestor():
    return DocumentIngestor()


# --------------------------------------------------------------------------- #
# Per-format loading
# --------------------------------------------------------------------------- #
def test_load_txt(ingestor):
    doc = ingestor.ingest(str(RAW / "hr_policy.txt"))
    assert doc.doc_type == "txt"
    assert "maternity leave" in doc.content.lower()
    assert doc.metadata["filename"] == "hr_policy.txt"
    assert doc.num_words > 100


def test_load_markdown(ingestor):
    doc = ingestor.ingest(str(RAW / "rag_overview.md"))
    assert doc.doc_type == "md"
    assert "Retrieval-Augmented Generation" in doc.content


def test_load_pdf(ingestor):
    doc = ingestor.ingest(str(RAW / "security_standard.pdf"))
    assert doc.doc_type == "pdf"
    assert doc.metadata["num_pages"] == 2
    assert "AES-256" in doc.content
    assert "\f" in doc.content  # page separator preserved
    assert doc.metadata.get("title") == "Acme Data Security Standard"


def test_load_docx(ingestor):
    doc = ingestor.ingest(str(RAW / "benefits_summary.docx"))
    assert doc.doc_type == "docx"
    assert "401(k)" in doc.content or "401k" in doc.content
    # Table content should be captured.
    assert "Employee Pays" in doc.content
    assert doc.metadata["num_tables"] == 1


# --------------------------------------------------------------------------- #
# Directory + batch
# --------------------------------------------------------------------------- #
def test_ingest_directory(ingestor):
    docs = ingestor.ingest_directory(str(RAW))
    types = {d.doc_type for d in docs}
    assert {"txt", "md", "pdf", "docx"}.issubset(types)
    assert len(docs) >= 4


def test_ingest_many_skips_errors(ingestor):
    docs = ingestor.ingest_many(
        [str(RAW / "hr_policy.txt"), str(RAW / "does_not_exist.txt")],
        skip_errors=True,
    )
    assert len(docs) == 1


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_missing_file_raises(ingestor):
    with pytest.raises(IngestionError):
        ingestor.ingest(str(RAW / "nope.txt"))


def test_unsupported_extension_raises(ingestor, tmp_path):
    weird = tmp_path / "data.xyz"
    weird.write_text("hello")
    with pytest.raises(UnsupportedFormatError):
        ingestor.ingest(str(weird))


def test_url_detection(ingestor):
    assert ingestor._is_url("https://example.com/page")
    assert ingestor._is_url("http://example.com")
    assert not ingestor._is_url("C:/local/file.txt")
    assert not ingestor._is_url("data/raw/x.txt")


def test_text_encoding_fallback(tmp_path):
    # Bytes that are invalid UTF-8 should still load via latin-1 fallback.
    p = tmp_path / "weird.txt"
    p.write_bytes("café résumé".encode("latin-1"))
    doc = TextLoader().load(str(p))
    assert "caf" in doc.content
