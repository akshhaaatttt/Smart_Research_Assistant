"""Tests for the foundation layer: config loading and shared schemas."""

import os

import pytest

from src.utils.config import AppConfig
from src.utils.exceptions import ConfigError
from src.utils.schemas import Chunk, Document, RAGResponse, RetrievedChunk


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_config_loads_defaults():
    cfg = AppConfig.load()
    assert cfg.get("project.name") == "Smart Research Assistant"
    assert cfg.get("chunking.chunk_size") == 800
    assert cfg.get("embeddings.active") == "minilm"


def test_dotted_access_and_default():
    cfg = AppConfig.load()
    assert cfg.get("embeddings.models.minilm.dimension") == 384
    assert cfg.get("nope.nope", "fallback") == "fallback"


def test_missing_key_without_default_raises():
    cfg = AppConfig.load()
    with pytest.raises(ConfigError):
        cfg.get("this.key.is.absent")


def test_env_var_expansion(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    cfg = AppConfig.load()
    assert cfg.get("generation.openai.base_url") == "http://localhost:1234/v1"


def test_env_var_default_when_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = AppConfig.load()
    assert cfg.get("generation.openai.base_url") == "https://api.openai.com/v1"


def test_overrides_do_not_mutate_original():
    cfg = AppConfig.load()
    overridden = cfg.with_overrides({"chunking": {"chunk_size": 1234}})
    assert overridden.get("chunking.chunk_size") == 1234
    assert cfg.get("chunking.chunk_size") == 800  # original untouched


def test_resolve_path_is_absolute():
    cfg = AppConfig.load()
    p = cfg.resolve_path("paths.data_raw")
    assert p.is_absolute()
    assert p.name == "raw"


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
def test_document_derives_stable_id():
    d1 = Document(content="abc", source="x.txt", doc_type="txt")
    d2 = Document(content="different", source="x.txt", doc_type="txt")
    # id is derived from source + type, so these match; it is deterministic.
    assert d1.doc_id == d2.doc_id
    assert len(d1.doc_id) == 16
    assert d1.num_words == 1


def test_chunk_metadata_roundtrip():
    ch = Chunk(content="hello there", source="x.txt", doc_id="doc1", chunk_index=2,
               metadata={"page": 5})
    meta = ch.to_metadata()
    assert meta["doc_id"] == "doc1"
    assert meta["chunk_index"] == 2
    assert meta["page"] == 5
    assert meta["chunk_id"] == ch.chunk_id


def test_rag_response_unique_sources_preserve_order():
    def mk(src, i):
        return RetrievedChunk(
            chunk=Chunk(content=f"c{i}", source=src, doc_id="d", chunk_index=i),
            score=1.0 - i * 0.1,
            rank=i,
        )

    resp = RAGResponse(
        query="q",
        answer="a",
        contexts=[mk("a.txt", 0), mk("b.txt", 1), mk("a.txt", 2)],
    )
    assert resp.sources == ["a.txt", "b.txt"]
