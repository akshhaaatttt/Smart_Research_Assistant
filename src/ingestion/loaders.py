"""Format-specific document loaders that produce a uniform Document from each input type."""

from __future__ import annotations

import abc
import datetime as _dt
from pathlib import Path
from typing import Any, Dict

from src.utils.exceptions import IngestionError
from src.utils.logger import get_logger
from src.utils.schemas import Document

logger = get_logger(__name__)


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


class BaseLoader(abc.ABC):
    """Abstract base class for all loaders."""

    #: File extension(s) this loader handles (lower-case, with dot). Empty for web.
    extensions: tuple[str, ...] = ()
    #: Short type label recorded on the produced Document.
    doc_type: str = "unknown"

    @abc.abstractmethod
    def load(self, source: str) -> Document:
        """Load ``source`` and return a populated :class:`Document`."""

    def _base_metadata(self, source: str) -> Dict[str, Any]:
        return {"doc_type": self.doc_type, "ingested_at": _now_iso()}


class TextLoader(BaseLoader):
    """Loader for plain-text and Markdown files."""

    extensions = (".txt", ".md")
    doc_type = "txt"

    def load(self, source: str) -> Document:
        path = Path(source)
        if not path.is_file():
            raise IngestionError(f"Text file not found: {source}")

        text = self._read_with_fallback(path)
        doc_type = "md" if path.suffix.lower() == ".md" else "txt"
        meta = self._base_metadata(source)
        meta.update(
            {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "doc_type": doc_type,
            }
        )
        logger.info("Loaded text file '%s' (%d chars)", path.name, len(text))
        return Document(content=text, source=str(path), doc_type=doc_type, metadata=meta)

    @staticmethod
    def _read_with_fallback(path: Path) -> str:
        """Read text trying UTF-8 first, then a lenient fallback."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        # Last resort: decode with replacement so ingestion never hard-fails.
        return path.read_bytes().decode("utf-8", errors="replace")


class PDFLoader(BaseLoader):
    """Loader for PDF files; preserves page boundaries as form-feed markers."""

    extensions = (".pdf",)
    doc_type = "pdf"

    def load(self, source: str) -> Document:
        from pypdf import PdfReader  # local import keeps module import cheap

        path = Path(source)
        if not path.is_file():
            raise IngestionError(f"PDF file not found: {source}")

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise IngestionError(f"Failed to open PDF '{source}': {exc}") from exc

        if reader.is_encrypted:
            # Empty password works for common view-only PDFs.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise IngestionError(f"PDF is encrypted and cannot be read: {source}") from exc

        pages: list[str] = []
        page_char_offsets: list[int] = []
        running = 0
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # a single bad page shouldn't kill the load
                logger.warning("Failed to extract a page from '%s': %s", path.name, exc)
                page_text = ""
            pages.append(page_text)
            page_char_offsets.append(running)
            running += len(page_text) + 1  # +1 for the form-feed separator

        content = "\f".join(pages)
        if not content.strip():
            raise IngestionError(
                f"No extractable text in PDF '{source}'. It may be a scanned image "
                f"(OCR is required but not enabled)."
            )

        meta = self._base_metadata(source)
        meta.update(
            {
                "filename": path.name,
                "extension": ".pdf",
                "size_bytes": path.stat().st_size,
                "num_pages": len(pages),
                "page_char_offsets": page_char_offsets,
            }
        )
        doc_info = getattr(reader, "metadata", None)
        if doc_info:
            if doc_info.title:
                meta["title"] = str(doc_info.title)
            if doc_info.author:
                meta["author"] = str(doc_info.author)

        logger.info("Loaded PDF '%s' (%d pages, %d chars)", path.name, len(pages), len(content))
        return Document(content=content, source=str(path), doc_type="pdf", metadata=meta)


class DocxLoader(BaseLoader):
    """Loader for Word .docx files; extracts paragraphs and table cells."""

    extensions = (".docx",)
    doc_type = "docx"

    def load(self, source: str) -> Document:
        import docx

        path = Path(source)
        if not path.is_file():
            raise IngestionError(f"DOCX file not found: {source}")

        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise IngestionError(f"Failed to open DOCX '{source}': {exc}") from exc

        parts: list[str] = [p.text for p in document.paragraphs if p.text and p.text.strip()]

        # python-docx keeps table content separate from paragraphs.
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        content = "\n".join(parts)
        if not content.strip():
            raise IngestionError(f"No extractable text in DOCX '{source}'.")

        meta = self._base_metadata(source)
        meta.update(
            {
                "filename": path.name,
                "extension": ".docx",
                "size_bytes": path.stat().st_size,
                "num_paragraphs": len(document.paragraphs),
                "num_tables": len(document.tables),
            }
        )
        core = document.core_properties
        if core.title:
            meta["title"] = core.title
        if core.author:
            meta["author"] = core.author

        logger.info("Loaded DOCX '%s' (%d chars)", path.name, len(content))
        return Document(content=content, source=str(path), doc_type="docx", metadata=meta)


class WebLoader(BaseLoader):
    """Loader for web pages: fetches HTML and extracts the main readable text."""

    extensions = ()
    doc_type = "web"

    def __init__(self, timeout: int = 20, user_agent: str = "SmartResearchAssistant/0.1",
                 max_bytes: int = 5_000_000) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_bytes = max_bytes

    def load(self, source: str) -> Document:
        import requests
        from bs4 import BeautifulSoup

        try:
            response = requests.get(
                source,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise IngestionError(f"Failed to fetch URL '{source}': {exc}") from exc

        raw = response.content[: self.max_bytes]
        soup = BeautifulSoup(raw, "html.parser")

        # Strip non-content elements so we keep only readable prose.
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else source
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        content = "\n".join(ln for ln in lines if ln)

        if not content.strip():
            raise IngestionError(f"No extractable text at URL '{source}'.")

        meta = self._base_metadata(source)
        meta.update({"url": source, "title": title, "status_code": response.status_code})
        logger.info("Loaded web page '%s' (%d chars)", source, len(content))
        return Document(content=content, source=source, doc_type="web", metadata=meta)
