"""Tests for the embeddings module.

These tests use the small MiniLM model (downloaded once and cached). They are the
first tests that require a model, so they double as an integration check that the
sentence-transformers stack works in this environment.
"""

import numpy as np
import pytest

from src.embeddings.embedder import EmbeddingManager, EmbeddingModel
from src.utils.exceptions import ConfigError, EmbeddingError


@pytest.fixture(scope="module")
def manager():
    return EmbeddingManager()


@pytest.fixture(scope="module")
def model(manager) -> EmbeddingModel:
    return manager.get_model("minilm")


def test_dimension_and_shape(model):
    assert model.dimension == 384
    vecs = model.embed_texts(["hello world", "goodbye world"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32


def test_vectors_are_normalized(model):
    vecs = model.embed_texts(["one", "two", "three"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_semantic_similarity_ordering(model):
    docs = model.embed_texts(
        [
            "Employees receive 26 weeks of paid maternity leave.",
            "The Eiffel Tower is located in Paris, France.",
        ]
    )
    q = model.embed_query("How long is maternity leave?")
    sims = docs @ q
    assert sims[0] > sims[1]  # policy sentence beats the unrelated one


def test_empty_inputs(model):
    assert model.embed_texts([]).shape == (0, 384)
    with pytest.raises(EmbeddingError):
        model.embed_query("   ")


def test_model_is_cached(manager):
    a = manager.get_model("minilm")
    b = manager.get_model("minilm")
    assert a is b  # same object, not reloaded


def test_unknown_model_raises(manager):
    with pytest.raises(ConfigError):
        manager.get_model("does-not-exist")


def test_available_models_from_config(manager):
    assert "minilm" in manager.available_models
    assert "mpnet" in manager.available_models
