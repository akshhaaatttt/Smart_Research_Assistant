"""Interchangeable text-splitting strategies (recursive, fixed, sentence) sharing a common contract."""

from __future__ import annotations

import abc
import re
from typing import List, Optional, Sequence

from src.utils.exceptions import ChunkingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseChunkStrategy(abc.ABC):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abc.abstractmethod
    def split(self, text: str) -> List[str]:
        """Split ``text`` into a list of non-empty chunk strings."""


class RecursiveChunkStrategy(BaseChunkStrategy):
    """Wraps LangChain's RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap)
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators) if separators else ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            keep_separator=True,
        )

    def split(self, text: str) -> List[str]:
        return [s for s in self._splitter.split_text(text) if s.strip()]


class FixedChunkStrategy(BaseChunkStrategy):
    """Fixed-width character windows with a fixed stride (size - overlap)."""

    def split(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        stride = self.chunk_size - self.chunk_overlap
        chunks: List[str] = []
        for start in range(0, len(text), stride):
            window = text[start : start + self.chunk_size]
            if window.strip():
                chunks.append(window.strip())
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunkStrategy(BaseChunkStrategy):
    """Greedily pack whole sentences up to the size budget, with sentence overlap."""

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def split(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self._SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            # A single over-budget sentence becomes its own chunk.
            if len(sentence) > self.chunk_size and not current:
                chunks.append(sentence)
                continue

            if current_len + len(sentence) + 1 > self.chunk_size and current:
                chunks.append(" ".join(current))
                current, current_len = self._overlap_tail(current)

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _overlap_tail(self, sentences: List[str]) -> tuple[List[str], int]:
        """Return the trailing sentences whose combined length ~ chunk_overlap."""
        tail: List[str] = []
        length = 0
        for sentence in reversed(sentences):
            if length + len(sentence) > self.chunk_overlap:
                break
            tail.insert(0, sentence)
            length += len(sentence) + 1
        return tail, length


def build_strategy(
    name: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[Sequence[str]] = None,
) -> BaseChunkStrategy:
    """Factory that returns the requested strategy instance."""
    name = (name or "recursive").lower()
    if name == "recursive":
        return RecursiveChunkStrategy(chunk_size, chunk_overlap, separators)
    if name == "fixed":
        return FixedChunkStrategy(chunk_size, chunk_overlap)
    if name == "sentence":
        return SentenceChunkStrategy(chunk_size, chunk_overlap)
    raise ChunkingError(
        f"Unknown chunking strategy '{name}'. Choose from: recursive, fixed, sentence."
    )
