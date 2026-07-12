"""Shared dataclasses that flow through the pipeline: Document -> Chunk ->
RetrievedChunk -> RAGResponse."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class Document:
    """A single source document after loading (before chunking)."""

    content: str
    source: str
    doc_type: str                     # pdf | txt | docx | md | web
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = _stable_id(self.source, self.doc_type)

    @property
    def num_chars(self) -> int:
        return len(self.content)

    @property
    def num_words(self) -> int:
        return len(self.content.split())


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    content: str
    source: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = _stable_id(self.doc_id, str(self.chunk_index), self.content[:64])

    @property
    def num_chars(self) -> int:
        return len(self.content)

    def to_metadata(self) -> Dict[str, Any]:
        """Flat metadata dict suitable for storing alongside a vector."""
        meta = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }
        meta.update(self.metadata)
        return meta


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever together with its similarity score."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def source(self) -> str:
        return self.chunk.source


@dataclass
class RAGResponse:
    """The full result of answering a query: answer + evidence + timings."""

    query: str
    answer: str
    contexts: List[RetrievedChunk] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> List[str]:
        """Unique source references, preserving retrieval order."""
        seen: List[str] = []
        for rc in self.contexts:
            if rc.source not in seen:
                seen.append(rc.source)
        return seen
