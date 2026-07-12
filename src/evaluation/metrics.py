"""Built-in, dependency-light RAG evaluation metrics.

Embedding-based cosine-similarity approximations of the RAGAS metrics, so evaluation
stays always-available, offline, and free when no LLM judge is present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.embeddings.embedder import EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_REL_THRESHOLD = 0.3  # similarity floor for "relevant"


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]


@dataclass
class EvalSample:
    """One evaluation datapoint."""

    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class SampleScores:
    faithfulness: float = float("nan")
    answer_relevancy: float = float("nan")
    context_precision: float = float("nan")
    context_recall: float = float("nan")
    answer_correctness: float = float("nan")

    def as_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_correctness": self.answer_correctness,
        }


class BuiltinEvaluator:
    """Compute embedding-based metric approximations."""

    def __init__(self, embedding_model: EmbeddingModel, rel_threshold: float = _REL_THRESHOLD):
        self.embedder = embedding_model
        self.rel_threshold = rel_threshold

    def evaluate_sample(self, sample: EvalSample) -> SampleScores:
        scores = SampleScores()
        answer_sents = _split_sentences(sample.answer)

        q_vec = self._embed_one(sample.question)
        a_vec = self._embed_one(sample.answer)
        ctx_mat = self._embed_many(sample.contexts)

        # faithfulness: answer sentences vs contexts.
        if answer_sents and ctx_mat.size:
            sent_mat = self._embed_many(answer_sents)
            sims = sent_mat @ ctx_mat.T             # (S, C)
            scores.faithfulness = float(np.mean(np.max(sims, axis=1)))

        # answer_relevancy: answer vs question.
        if q_vec is not None and a_vec is not None:
            scores.answer_relevancy = float(np.clip(a_vec @ q_vec, 0.0, 1.0))

        # context_precision: rank-weighted relevance of contexts to the question.
        if q_vec is not None and ctx_mat.size:
            ctx_sims = ctx_mat @ q_vec              # (C,)
            relevant = ctx_sims >= self.rel_threshold
            scores.context_precision = self._rank_weighted_precision(relevant)

        # context_recall: ground-truth coverage by contexts.
        if sample.ground_truth and ctx_mat.size:
            gt_sents = _split_sentences(sample.ground_truth)
            if gt_sents:
                gt_mat = self._embed_many(gt_sents)
                cover = np.max(gt_mat @ ctx_mat.T, axis=1)   # best ctx per GT sentence
                scores.context_recall = float(np.mean(cover >= self.rel_threshold))

        # answer_correctness: generated answer vs ground truth.
        if sample.ground_truth and a_vec is not None:
            gt_vec = self._embed_one(sample.ground_truth)
            if gt_vec is not None:
                scores.answer_correctness = float(np.clip(a_vec @ gt_vec, 0.0, 1.0))

        return scores

    @staticmethod
    def _rank_weighted_precision(relevant: np.ndarray) -> float:
        """Average precision@k style score rewarding relevant chunks ranked higher."""
        if relevant.size == 0:
            return float("nan")
        hits = 0
        weighted = 0.0
        for i, is_rel in enumerate(relevant, start=1):
            if is_rel:
                hits += 1
                weighted += hits / i
        return float(weighted / hits) if hits else 0.0

    def _embed_one(self, text: Optional[str]) -> Optional[np.ndarray]:
        if not text or not text.strip():
            return None
        return self.embedder.embed_texts([text])[0]

    def _embed_many(self, texts: List[str]) -> np.ndarray:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return np.empty((0, self.embedder.dimension), dtype=np.float32)
        return self.embedder.embed_texts(texts)


@dataclass
class EvaluationReport:
    """Aggregated results over a set of samples."""

    per_sample: List[Dict[str, float]] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)
    method: str = "builtin"

    def to_dict(self) -> Dict:
        return {"method": self.method, "aggregate": self.aggregate, "per_sample": self.per_sample}


def aggregate_scores(sample_dicts: List[Dict[str, float]], method: str = "builtin") -> EvaluationReport:
    """Mean each metric across samples, ignoring NaNs (missing ground truth etc.)."""
    if not sample_dicts:
        return EvaluationReport(method=method)
    metrics = sample_dicts[0].keys()
    aggregate: Dict[str, float] = {}
    for metric in metrics:
        values = [d[metric] for d in sample_dicts if not np.isnan(d.get(metric, float("nan")))]
        aggregate[metric] = float(np.mean(values)) if values else float("nan")
    return EvaluationReport(per_sample=sample_dicts, aggregate=aggregate, method=method)
