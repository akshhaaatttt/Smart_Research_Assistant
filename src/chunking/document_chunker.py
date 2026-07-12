"""Turn cleaned Documents into retrievable Chunks, splitting per page so chunks carry page numbers."""

from __future__ import annotations

from typing import List, Optional

from src.chunking.strategies import build_strategy
from src.utils.config import AppConfig
from src.utils.exceptions import ChunkingError
from src.utils.logger import get_logger
from src.utils.schemas import Chunk, Document

logger = get_logger(__name__)

_PAGE_SEP = "\f"


class DocumentChunker:
    """Split Documents into Chunks according to the configured strategy."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.load()
        chunk_cfg = self.config.section("chunking")

        self.strategy_name: str = chunk_cfg.get("strategy", "recursive")
        self.chunk_size: int = chunk_cfg.get("chunk_size", 800)
        self.chunk_overlap: int = chunk_cfg.get("chunk_overlap", 120)
        self.min_chunk_chars: int = chunk_cfg.get("min_chunk_chars", 50)
        separators = chunk_cfg.get("separators")

        self._strategy = build_strategy(
            self.strategy_name, self.chunk_size, self.chunk_overlap, separators
        )

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a single Document into a list of Chunks with page metadata."""
        if not document.content or not document.content.strip():
            logger.warning("Document '%s' has no content to chunk", document.source)
            return []

        pages = document.content.split(_PAGE_SEP)
        has_pages = len(pages) > 1

        chunks: List[Chunk] = []
        index = 0
        for page_number, page_text in enumerate(pages, start=1):
            for piece in self._strategy.split(page_text):
                piece = piece.strip()
                if len(piece) < self.min_chunk_chars:
                    continue
                metadata = {
                    "doc_type": document.doc_type,
                    "strategy": self.strategy_name,
                }
                if has_pages:
                    metadata["page"] = page_number
                # Carry through document-level metadata useful for citation.
                for key in ("filename", "title", "url"):
                    if key in document.metadata:
                        metadata[key] = document.metadata[key]

                chunks.append(
                    Chunk(
                        content=piece,
                        source=document.source,
                        doc_id=document.doc_id,
                        chunk_index=index,
                        metadata=metadata,
                    )
                )
                index += 1

        if not chunks:
            raise ChunkingError(
                f"Chunking produced no chunks for '{document.source}'. "
                f"Check chunk_size ({self.chunk_size}) vs. min_chunk_chars "
                f"({self.min_chunk_chars})."
            )

        logger.info(
            "Chunked '%s' into %d chunk(s) [strategy=%s, size=%d, overlap=%d]",
            document.metadata.get("filename", document.source),
            len(chunks),
            self.strategy_name,
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk many Documents into a single flat list of Chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Produced %d chunk(s) from %d document(s)", len(all_chunks), len(documents))
        return all_chunks
