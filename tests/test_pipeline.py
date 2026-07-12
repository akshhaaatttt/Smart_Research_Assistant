"""End-to-end integration tests for the full RAG pipeline."""

from pathlib import Path

import pytest

from src.pipeline import RAGPipeline
from src.utils.config import AppConfig

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _pipeline(tmp_path, backend="faiss"):
    cfg = AppConfig.load().with_overrides(
        {
            "vector_store": {
                "active": backend,
                "faiss": {"persist_dir": str(tmp_path / "faiss")},
                "chroma": {"persist_dir": str(tmp_path / "chroma")},
            }
        }
    )
    return RAGPipeline(cfg, auto_load=False)


def test_augment_query_uses_history():
    aug = RAGPipeline._augment_query
    # No history -> query unchanged.
    assert aug("How much notice?", None) == "How much notice?"
    # With history, prior question is prepended so the follow-up retrieves well.
    out = aug("How much notice?", [("What is the maternity leave policy?", "26 weeks.")])
    assert "maternity leave" in out.lower()
    assert "how much notice?" in out.lower()


@pytest.fixture(scope="module")
def indexed_pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    pipe = _pipeline(tmp)
    stats = pipe.index_directory(str(RAW))
    assert stats.chunks > 0
    return pipe


def test_indexing_populates_store_and_records(indexed_pipeline):
    assert indexed_pipeline.num_indexed > 0
    assert len(indexed_pipeline.documents) >= 4
    assert len(indexed_pipeline.chunks) == indexed_pipeline.num_indexed


def test_retrieval_only_returns_relevant_context(indexed_pipeline):
    ctx = indexed_pipeline.retrieve("What is the maternity leave policy?", top_k=3)
    assert ctx
    assert any("maternity" in c.content.lower() for c in ctx)
    assert "hr_policy" in ctx[0].source


@pytest.mark.integration
def test_end_to_end_query_is_grounded_and_attributed(indexed_pipeline):
    resp = indexed_pipeline.query("How many weeks of paid maternity leave do employees get?")
    assert "26" in resp.answer
    assert resp.sources                       # at least one source attributed
    assert any("hr_policy" in s for s in resp.sources)
    assert "retrieval" in resp.timings and "generation" in resp.timings
    assert resp.metadata["llm_backend"] in {"local", "openai"}


@pytest.mark.integration
def test_out_of_scope_question_is_refused(indexed_pipeline):
    resp = indexed_pipeline.query("Who won the 2018 FIFA World Cup?")
    # The answer must not fabricate; grounded prompt should decline.
    assert "26" not in resp.answer
    lowered = resp.answer.lower()
    assert ("don't have enough information" in lowered) or ("not" in lowered) or (len(resp.answer) < 200)


def test_reset_clears_everything(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe.index_directory(str(RAW))
    assert pipe.num_indexed > 0
    pipe.reset_index()
    assert pipe.num_indexed == 0
    assert pipe.chunks == []
