"""Tests for the preprocessing / cleaning module."""

from pathlib import Path

import pytest

from src.ingestion.document_loader import DocumentIngestor
from src.preprocessing.cleaner import DocumentPreprocessor, TextCleaner
from src.utils.schemas import Document

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner()


def test_hyphenation_repair(cleaner):
    out = cleaner.clean("This is encryp-\ntion at rest.")
    assert "encryption" in out
    assert "encryp-" not in out


def test_smart_quotes_folded(cleaner):
    out = cleaner.clean("The “leave” policy’s terms…")
    assert '"leave"' in out
    assert "policy's" in out
    assert "..." in out


def test_whitespace_collapsed(cleaner):
    out = cleaner.clean("word1     word2\t\tword3\n\n\n\n\nword4")
    assert "word1 word2 word3" in out
    assert "\n\n\n" not in out


def test_page_number_lines_removed(cleaner):
    text = "Real content line.\n12\nPage 3\nMore content."
    out = cleaner.clean(text)
    assert "Real content line." in out
    assert "More content." in out
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert "12" not in lines
    assert "Page 3" not in lines


def test_page_separator_preserved(cleaner):
    out = cleaner.clean("Page one text.\fPage two text.")
    assert "\f" in out
    assert out.split("\f")[0].strip() == "Page one text."


def test_control_chars_removed(cleaner):
    out = cleaner.clean("hello\x00\x07world")
    assert out == "helloworld"


def test_empty_input(cleaner):
    assert cleaner.clean("") == ""


def test_document_preprocessor_records_stats():
    doc = Document(content="word1     word2\n\n\n\nword3", source="x.txt", doc_type="txt")
    cleaned = DocumentPreprocessor().process(doc)
    assert cleaned.doc_id == doc.doc_id  # identity preserved
    assert cleaned.metadata["cleaned"] is True
    assert cleaned.metadata["chars_before_cleaning"] == len(doc.content)
    assert cleaned.metadata["chars_after_cleaning"] == len(cleaned.content)
    assert "word1 word2" in cleaned.content


def test_running_headers_and_footers_removed(cleaner):
    """Lines repeated across pages (journal header + page-numbered footer) are dropped."""
    bodies = [
        "The convolutional layers extract spatial features from images.",
        "Pruning reduces the number of parameters in the network.",
        "Quantization maps weights to fixed-point representations.",
        "The FPGA implementation improves inference latency substantially.",
        "Ultrasound images are preprocessed before classification.",
        "The android deployment enables point-of-care screening.",
    ]
    pages = [
        f"A. Author et al.: Deep Learning Paper\n{body}\nVOLUME 12, 2024 {179190 + i}"
        for i, body in enumerate(bodies)
    ]
    text = "\f".join(pages)
    out = cleaner.clean(text)
    assert "A. Author et al." not in out          # constant running header removed
    assert "VOLUME 12, 2024" not in out            # digit-varying footer removed
    assert "quantization maps weights" in out.lower()  # unique body preserved


def test_short_docs_keep_repeated_lines(cleaner):
    """With fewer than 3 pages there is no reliable running-header signal."""
    text = "Header line\fHeader line"  # only 2 pages
    out = cleaner.clean(text)
    assert "Header line" in out


def test_end_to_end_pdf_cleaning():
    """A real PDF should retain its substantive content after cleaning."""
    doc = DocumentIngestor().ingest(str(RAW / "security_standard.pdf"))
    cleaned = DocumentPreprocessor().process(doc)
    assert "AES-256" in cleaned.content
    assert "\f" in cleaned.content  # page structure retained for citation
