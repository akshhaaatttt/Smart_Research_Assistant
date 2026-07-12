"""Logging setup. Modules get their logger via ``get_logger``; output goes to the
console and a rotating file under ``logs/``."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"

_configured = False


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: str = "rag_system.log",
) -> None:
    """Configure the root logger once. Repeated calls are no-ops."""
    global _configured
    if _configured:
        return

    log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in ("urllib3", "httpx", "sentence_transformers", "chromadb", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, ensuring logging is configured first."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
