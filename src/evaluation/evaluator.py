"""Unified RAG evaluator.

Single :class:`RAGEvaluator` interface that picks RAGAS when configured and falls back
to the built-in embedding metrics, and can build eval samples straight from the pipeline.
"""

from __future__ import annotations

from typing import List, Optional

from src.embeddings.embedder import EmbeddingModel
from src.evaluation.metrics import (
    BuiltinEvaluator,
    EvalSample,
    EvaluationReport,
    aggregate_scores,
)
from src.evaluation.ragas_eval import RagasEvaluator
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or AppConfig.load()
        self.builtin = BuiltinEvaluator(embedding_model)
        self.ragas = RagasEvaluator(self.config, embedding_model=embedding_model)
        self._use_ragas = self.config.get("evaluation.use_ragas", True)

    def method_in_use(self) -> str:
        if self._use_ragas and self.ragas.is_configured():
            return "ragas"
        return "builtin"

    def evaluate(self, samples: List[EvalSample]) -> EvaluationReport:
        """Score the samples with the best available method."""
        if not samples:
            return EvaluationReport(method=self.method_in_use())

        if self.method_in_use() == "ragas":
            try:
                logger.info("Evaluating %d sample(s) with RAGAS", len(samples))
                return self.ragas.evaluate(samples)
            except Exception as exc:
                logger.warning("RAGAS failed (%s); falling back to built-in metrics.", exc)

        logger.info("Evaluating %d sample(s) with built-in metrics", len(samples))
        per_sample = [self.builtin.evaluate_sample(s).as_dict() for s in samples]
        return aggregate_scores(per_sample, method="builtin")

    def evaluate_pipeline(
        self,
        pipeline,
        questions: List[str],
        ground_truths: Optional[List[Optional[str]]] = None,
        top_k: Optional[int] = None,
    ) -> EvaluationReport:
        """Run each question through ``pipeline`` and evaluate the outcomes."""
        ground_truths = ground_truths or [None] * len(questions)
        if len(ground_truths) != len(questions):
            raise ValueError("questions and ground_truths must be the same length.")

        samples: List[EvalSample] = []
        for question, gt in zip(questions, ground_truths):
            response = pipeline.query(question, top_k=top_k)
            samples.append(
                EvalSample(
                    question=question,
                    answer=response.answer,
                    contexts=[rc.content for rc in response.contexts],
                    ground_truth=gt,
                )
            )
        return self.evaluate(samples)
