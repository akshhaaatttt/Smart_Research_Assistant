"""Environment bootstrap. Import before any transformers/sentence_transformers use.

The local Anaconda env ships TensorFlow + Keras 3, which breaks HuggingFace
``transformers`` on import. Forcing the PyTorch-only path (``USE_TF=0``) fixes the
crash and skips loading TensorFlow entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULTS = {
    "USE_TF": "0",
    "USE_FLAX": "0",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def _load_dotenv() -> None:
    """Load a project-root ``.env`` if present. Real environment variables win."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    except OSError:
        pass


def configure_environment() -> None:
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, value)
    _load_dotenv()


configure_environment()
