"""Configurable text cleaning and normalization that preserves page boundaries for chunking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Optional

from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.schemas import Document

logger = get_logger(__name__)

# Fold unicode punctuation to ASCII so smart quotes don't derail retrieval.
_PUNCTUATION_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
    "•": "-", "﻿": "",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCTUATION_MAP)))

# A line that is just a page number, e.g. "12" or "Page 12" or "- 12 -".
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.IGNORECASE)

# Word split across a line break by hyphenation: "encryp-\ntion" -> "encryption".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

# Control characters except tab/newline/form-feed.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")

_PAGE_SEP = "\f"


class TextCleaner:
    """Apply a configurable sequence of normalization steps to raw text."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        cfg = config or AppConfig.load()
        pre = cfg.section("preprocessing")
        self.lowercase: bool = pre.get("lowercase", False)
        self.normalize_unicode: bool = pre.get("normalize_unicode", True)
        self.collapse_whitespace: bool = pre.get("collapse_whitespace", True)
        self.strip_boilerplate: bool = pre.get("strip_boilerplate", True)
        self.min_paragraph_chars: int = pre.get("min_paragraph_chars", 3)

    def clean(self, text: str) -> str:
        """Return a cleaned copy of ``text``, preserving page separators."""
        if not text:
            return ""
        # Clean each page independently so \f boundaries survive intact.
        pages = text.split(_PAGE_SEP)
        running = self._detect_running_lines(pages) if self.strip_boilerplate else set()
        cleaned_pages = [self._clean_segment(p, running) for p in pages]
        cleaned_pages = [p for p in cleaned_pages if p] or [""]
        return _PAGE_SEP.join(cleaned_pages)

    @staticmethod
    def _normalize_line(line: str) -> str:
        """Digit-insensitive form so footers with varying page numbers match."""
        return re.sub(r"\d+", "#", line.strip())

    @classmethod
    def _detect_running_lines(cls, pages: list[str]) -> set[str]:
        """Find short lines that recur across many pages (running headers/footers).

        Matching uses a digit-normalized form so footers whose page number changes
        each page still match. A line counts as boilerplate if it's short and recurs
        on at least ~40% of pages (min 3).
        """
        if len(pages) < 3:
            return set()
        from collections import Counter

        counts: Counter = Counter()
        for page in pages:
            seen_on_page = {
                cls._normalize_line(ln) for ln in page.split("\n")
                if 3 < len(ln.strip()) <= 100
            }
            counts.update(seen_on_page)
        threshold = max(3, int(0.4 * len(pages)))
        return {line for line, n in counts.items() if n >= threshold}

    def _clean_segment(self, text: str, running_lines: set[str] = frozenset()) -> str:
        # Repair hyphenated line breaks before we touch newlines.
        text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
            text = _PUNCT_RE.sub(lambda m: _PUNCTUATION_MAP[m.group()], text)

        text = _CONTROL_RE.sub("", text)

        lines = text.split("\n")
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if self.strip_boilerplate and _PAGE_NUMBER_RE.match(stripped):
                continue
            if self._normalize_line(stripped) in running_lines:  # running header/footer
                continue
            kept.append(line)
        text = "\n".join(kept)

        if self.collapse_whitespace:
            text = self._collapse(text)

        if self.strip_boilerplate and len(text.strip()) < self.min_paragraph_chars:
            return ""

        if self.lowercase:
            text = text.lower()

        return text.strip()

    @staticmethod
    def _collapse(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)  # runs of spaces/tabs, but keep newlines
        text = re.sub(r" *\n", "\n", text)  # trailing spaces per line
        text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines -> paragraph break
        return text


class DocumentPreprocessor:
    """Clean whole :class:`Document`s, recording before/after stats in metadata."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.load()
        self.cleaner = TextCleaner(self.config)

    def process(self, document: Document) -> Document:
        """Return a new Document with cleaned content and enriched metadata."""
        original_chars = len(document.content)
        cleaned = self.cleaner.clean(document.content)

        metadata = dict(document.metadata)
        metadata.update(
            {
                "cleaned": True,
                "chars_before_cleaning": original_chars,
                "chars_after_cleaning": len(cleaned),
            }
        )
        logger.info(
            "Cleaned '%s': %d -> %d chars (%.1f%% retained)",
            document.metadata.get("filename", document.source),
            original_chars,
            len(cleaned),
            100.0 * len(cleaned) / original_chars if original_chars else 0.0,
        )
        # replace() keeps doc_id so cleaning is transparent to the pipeline.
        return replace(document, content=cleaned, metadata=metadata)

    def process_many(self, documents: list[Document]) -> list[Document]:
        return [self.process(doc) for doc in documents]
