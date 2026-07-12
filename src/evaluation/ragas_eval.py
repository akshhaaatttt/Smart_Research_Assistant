"""RAGAS integration (optional, LLM-judged evaluation).

Preferred evaluator when an OpenAI-compatible API key is available; imported lazily
and guarded so failures fall back to the built-in metrics. Includes a compatibility
shim for recent RAGAS releases that still import the relocated
``langchain_community.chat_models.vertexai`` module (unused here, so a stub suffices).
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import List, Optional

from src.embeddings.embedder import EmbeddingModel
from src.evaluation.metrics import EvalSample, EvaluationReport, aggregate_scores
from src.utils.config import AppConfig
from src.utils.exceptions import EvaluationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_local_embeddings(embedding_model: EmbeddingModel):
    """Wrap our Sentence-Transformers model in the LangChain Embeddings interface.

    Lets RAGAS run embedding metrics locally, so evaluation works with chat providers
    that have no embeddings endpoint (e.g. Groq) and stays in the retrieval vector space.
    """
    from langchain_core.embeddings import Embeddings

    class _LocalEmbeddings(Embeddings):
        def __init__(self, model: EmbeddingModel):
            self._model = model

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self._model.embed_texts(list(texts)).tolist()

        def embed_query(self, text: str) -> List[float]:
            return self._model.embed_query(text).tolist()

    return _LocalEmbeddings(embedding_model)


def _install_vertexai_shim() -> None:
    """Provide a stub for the relocated langchain vertexai chat model if missing."""
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:
        importlib.import_module(mod_name)
        return  # real module exists; no shim needed
    except Exception:
        pass
    try:
        stub = types.ModuleType(mod_name)

        class _Unavailable:  # placeholder; never instantiated on the OpenAI path
            def __init__(self, *args, **kwargs):
                raise RuntimeError("ChatVertexAI is not available in this environment.")

        stub.ChatVertexAI = _Unavailable  # type: ignore[attr-defined]
        sys.modules[mod_name] = stub
        logger.debug("Installed langchain vertexai compatibility shim for RAGAS.")
    except Exception as exc:  # pragma: no cover
        logger.debug("Could not install vertexai shim: %s", exc)


def ragas_available() -> bool:
    """Return True if RAGAS and the OpenAI langchain bindings can be imported."""
    _install_vertexai_shim()
    try:
        importlib.import_module("ragas")
        importlib.import_module("langchain_openai")
        return True
    except Exception as exc:
        logger.info("RAGAS not available (%s); built-in metrics will be used.", exc)
        return False


class RagasEvaluator:
    """Evaluate samples with RAGAS using an OpenAI-compatible LLM + embeddings."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.config = config or AppConfig.load()
        self.embedding_model = embedding_model

    def is_configured(self) -> bool:
        api_key = (self.config.get("generation.openai.api_key", "") or "").strip()
        return bool(api_key) and ragas_available()

    def evaluate(self, samples: List[EvalSample]) -> EvaluationReport:
        if not samples:
            return EvaluationReport(method="ragas")
        if not self.is_configured():
            raise EvaluationError(
                "RAGAS requires an OpenAI-compatible API key and a working RAGAS install."
            )

        _install_vertexai_shim()
        try:
            from datasets import Dataset
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                answer_correctness,
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except Exception as exc:
            raise EvaluationError(f"Failed to import RAGAS stack: {exc}") from exc

        openai_cfg = self.config.section("generation").get("openai", {})
        api_key = openai_cfg.get("api_key")
        base_url = openai_cfg.get("base_url") or None
        judge = ChatOpenAI(model=openai_cfg.get("model", "gpt-4o-mini"),
                           api_key=api_key, base_url=base_url, temperature=0.0)

        # Prefer local embeddings so RAGAS works with chat providers that have no
        # embeddings endpoint (e.g. Groq). Fall back to the provider's embeddings.
        if self.embedding_model is not None:
            embeddings = _build_local_embeddings(self.embedding_model)
        else:
            embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url)

        has_gt = all(s.ground_truth for s in samples)
        metrics = [faithfulness, answer_relevancy, context_precision]
        if has_gt:
            metrics += [context_recall, answer_correctness]

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
        }
        if has_gt:
            data["ground_truth"] = [s.ground_truth for s in samples]
        dataset = Dataset.from_dict(data)

        try:
            result = ragas_evaluate(dataset, metrics=metrics, llm=judge, embeddings=embeddings)
            df = result.to_pandas()
        except Exception as exc:
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

        metric_cols = [c for c in df.columns if c in {
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "answer_correctness",
        }]
        per_sample = [
            {m: float(row[m]) for m in metric_cols}
            for _, row in df.iterrows()
        ]
        report = aggregate_scores(per_sample, method="ragas")
        return report
