"""Document ingestion facade routing file paths, directories, and URLs to the right loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from src.ingestion.loaders import (
    BaseLoader,
    DocxLoader,
    PDFLoader,
    TextLoader,
    WebLoader,
)
from src.utils.config import AppConfig
from src.utils.exceptions import IngestionError, UnsupportedFormatError
from src.utils.logger import get_logger
from src.utils.schemas import Document

logger = get_logger(__name__)


class DocumentIngestor:
    """Route inputs to the appropriate loader and return uniform Documents."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.load()

        web_cfg = self.config.get("ingestion.web", {})
        self._web_loader = WebLoader(
            timeout=web_cfg.get("timeout_seconds", 20),
            user_agent=web_cfg.get("user_agent", "SmartResearchAssistant/0.1"),
            max_bytes=web_cfg.get("max_content_bytes", 5_000_000),
        )

        self._registry: Dict[str, BaseLoader] = {}
        for loader in (TextLoader(), PDFLoader(), DocxLoader()):
            for ext in loader.extensions:
                self._registry[ext] = loader

        self._supported = set(self.config.get("ingestion.supported_extensions", []))

    @property
    def supported_extensions(self) -> List[str]:
        return sorted(self._registry.keys())

    def ingest(self, source: str) -> Document:
        """Ingest a single source (file path or URL)."""
        if self._is_url(source):
            return self._web_loader.load(source)
        return self._ingest_file(source)

    def ingest_many(self, sources: List[str], *, skip_errors: bool = True) -> List[Document]:
        """Ingest multiple sources, optionally skipping (and logging) failures."""
        documents: List[Document] = []
        for src in sources:
            try:
                documents.append(self.ingest(src))
            except IngestionError as exc:
                if not skip_errors:
                    raise
                logger.error("Skipping '%s': %s", src, exc)
        logger.info("Ingested %d/%d source(s)", len(documents), len(sources))
        return documents

    def ingest_directory(
        self, directory: str, *, recursive: bool = True, skip_errors: bool = True
    ) -> List[Document]:
        """Ingest every supported file in a directory."""
        root = Path(directory)
        if not root.is_dir():
            raise IngestionError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        candidates = [
            p for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in self._registry
        ]
        logger.info("Found %d ingestible file(s) under '%s'", len(candidates), directory)
        return self.ingest_many([str(p) for p in candidates], skip_errors=skip_errors)

    def _ingest_file(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"File not found: {source}")

        ext = path.suffix.lower()
        loader = self._registry.get(ext)
        if loader is None:
            raise UnsupportedFormatError(
                f"Unsupported file type '{ext}'. Supported: {self.supported_extensions}"
            )
        return loader.load(source)

    @staticmethod
    def _is_url(source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
