"""Embedding generation.

Wraps ``sentence-transformers`` models behind a small, uniform interface with
per-process model caching and unit normalization (so inner product == cosine).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.utils.config import AppConfig
from src.utils.exceptions import ConfigError, EmbeddingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keyed by (name, device); shared across instances so Streamlit reruns don't
# reload the same weights and exhaust memory.
_ST_MODEL_CACHE: dict = {}


def _resolve_device(requested: str) -> str:
    """Resolve the 'auto' device to cuda when available, else cpu."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch always present here
        return "cpu"


class EmbeddingModel:
    """A single loaded embedding model with a uniform encode interface."""

    def __init__(
        self,
        key: str,
        model_name: str,
        *,
        expected_dim: Optional[int] = None,
        device: str = "auto",
        normalize: bool = True,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.key = key
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.normalize = normalize
        self.batch_size = batch_size

        logger.info("Loading embedding model '%s' (%s) on %s", key, model_name, self.device)
        self._model = self._load_model(model_name, cache_dir)

        self.dimension = self._model.get_sentence_embedding_dimension()
        if expected_dim is not None and expected_dim != self.dimension:
            logger.warning(
                "Configured dimension for '%s' (%d) != actual (%d); using actual.",
                key, expected_dim, self.dimension,
            )

    def _load_model(self, model_name: str, cache_dir):
        """Load the SentenceTransformer, working around the 'meta tensor' failure.

        Low-memory/threaded environments can leave the model on the ``meta`` device
        and fail with "Cannot copy out of meta tensor"; retrying with
        ``low_cpu_mem_usage=False`` forces eager weight materialization.
        """
        from sentence_transformers import SentenceTransformer

        import gc

        cache_key = (model_name, self.device)
        if cache_key in _ST_MODEL_CACHE:
            logger.info("Reusing cached embedding model '%s'", model_name)
            return _ST_MODEL_CACHE[cache_key]

        # Free transient garbage before the allocation-heavy load; on tight machines
        # a meta-tensor failure here is really an out-of-memory symptom.
        gc.collect()

        try:
            model = SentenceTransformer(model_name, device=self.device, cache_folder=cache_dir)
            _ST_MODEL_CACHE[cache_key] = model
            return model
        except NotImplementedError as exc:
            if "meta tensor" not in str(exc).lower():
                raise EmbeddingError(f"Failed to load embedding model '{model_name}': {exc}") from exc
            logger.warning(
                "Meta-tensor load for '%s' failed (low memory); freeing memory and retrying "
                "with eager weight loading.", model_name,
            )
            gc.collect()
            try:
                model = SentenceTransformer(
                    model_name, device=self.device, cache_folder=cache_dir,
                    model_kwargs={"low_cpu_mem_usage": False},
                )
                _ST_MODEL_CACHE[cache_key] = model
                return model
            except Exception as exc2:
                raise EmbeddingError(
                    f"Could not load embedding model '{model_name}': the system ran out of "
                    f"memory while materializing the model weights. Close other applications "
                    f"to free RAM and try again. (underlying error: {exc2})"
                ) from exc2
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed a list of documents/chunks. Returns an (N, dim) float32 array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a 1-D (dim,) float32 array."""
        if not query or not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self.embed_texts([query])[0]

    def __repr__(self) -> str:  # pragma: no cover
        return f"EmbeddingModel(key={self.key!r}, dim={self.dimension}, device={self.device!r})"


class EmbeddingManager:
    """Load, cache, and switch between embedding models defined in config."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.load()
        self._models: Dict[str, EmbeddingModel] = {}
        self._active_key: str = self.config.get("embeddings.active")

    @property
    def available_models(self) -> List[str]:
        return list(self.config.get("embeddings.models").keys())

    @property
    def active_key(self) -> str:
        return self._active_key

    def get_model(self, key: Optional[str] = None) -> EmbeddingModel:
        """Return a (cached) EmbeddingModel for ``key`` (default: active model)."""
        key = key or self._active_key
        if key in self._models:
            return self._models[key]

        models_cfg = self.config.get("embeddings.models")
        if key not in models_cfg:
            raise ConfigError(
                f"Unknown embedding model '{key}'. Available: {self.available_models}"
            )
        spec = models_cfg[key]

        model = EmbeddingModel(
            key=key,
            model_name=spec["model_name"],
            expected_dim=spec.get("dimension"),
            device=self.config.get("embeddings.device", "auto"),
            normalize=self.config.get("embeddings.normalize", True),
            batch_size=self.config.get("embeddings.batch_size", 32),
            cache_dir=str(self.config.resolve_path("embeddings.cache_dir")),
        )
        self._models[key] = model
        return model

    def set_active(self, key: str) -> EmbeddingModel:
        """Switch the active model, loading it if necessary."""
        model = self.get_model(key)
        self._active_key = key
        logger.info("Active embedding model set to '%s'", key)
        return model
