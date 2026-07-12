"""Tests for the evaluation module (built-in metrics + RAGAS fallback wiring)."""

import math

import pytest

from src.embeddings.embedder import EmbeddingManager
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.metrics import BuiltinEvaluator, EvalSample, aggregate_scores
from src.evaluation.ragas_eval import RagasEvaluator, ragas_available
from src.utils.config import AppConfig

CONTEXTS = [
    "Employees are entitled to 26 weeks of paid maternity leave.",
    "To qualify, an employee must have completed 12 months of continuous service.",
]


@pytest.fixture(scope="module")
def evaluator():
    return BuiltinEvaluator(EmbeddingManager().get_model("minilm"))


def test_grounded_answer_scores_high_faithfulness(evaluator):
    good = EvalSample(
        question="How long is maternity leave?",
        answer="Employees get 26 weeks of paid maternity leave.",
        contexts=CONTEXTS,
        ground_truth="Employees are entitled to 26 weeks of paid maternity leave.",
    )
    s = evaluator.evaluate_sample(good)
    assert s.faithfulness > 0.6
    assert s.answer_relevancy > 0.4
    assert s.answer_correctness > 0.7
    assert 0.0 <= s.context_precision <= 1.0


def test_hallucinated_answer_scores_lower_faithfulness(evaluator):
    grounded = evaluator.evaluate_sample(
        EvalSample("How long is maternity leave?",
                   "Employees get 26 weeks of paid maternity leave.", CONTEXTS)
    )
    hallucinated = evaluator.evaluate_sample(
        EvalSample("How long is maternity leave?",
                   "The Great Wall of China is visible from space and made of cheese.",
                   CONTEXTS)
    )
    assert hallucinated.faithfulness < grounded.faithfulness


def test_offtopic_answer_scores_lower_relevancy(evaluator):
    on = evaluator.evaluate_sample(
        EvalSample("How long is maternity leave?",
                   "Maternity leave is 26 weeks.", CONTEXTS)
    )
    off = evaluator.evaluate_sample(
        EvalSample("How long is maternity leave?",
                   "Photosynthesis converts sunlight into chemical energy.", CONTEXTS)
    )
    assert off.answer_relevancy < on.answer_relevancy


def test_context_recall_requires_ground_truth(evaluator):
    without = evaluator.evaluate_sample(
        EvalSample("q", "a", CONTEXTS)  # no ground truth
    )
    assert math.isnan(without.context_recall)
    with_gt = evaluator.evaluate_sample(
        EvalSample("q", "a", CONTEXTS, ground_truth="Employees get 26 weeks maternity leave.")
    )
    assert not math.isnan(with_gt.context_recall)


def test_aggregate_ignores_nan():
    per_sample = [
        {"faithfulness": 0.8, "context_recall": float("nan")},
        {"faithfulness": 0.6, "context_recall": 0.5},
    ]
    report = aggregate_scores(per_sample)
    assert report.aggregate["faithfulness"] == pytest.approx(0.7)
    assert report.aggregate["context_recall"] == pytest.approx(0.5)  # NaN dropped


def test_unified_evaluator_uses_builtin_without_key():
    cfg = AppConfig.load().with_overrides({"generation": {"openai": {"api_key": ""}}})
    ev = RAGEvaluator(EmbeddingManager().get_model("minilm"), cfg)
    assert ev.method_in_use() == "builtin"
    report = ev.evaluate([EvalSample("q", "Maternity leave is 26 weeks.", CONTEXTS)])
    assert report.method == "builtin"
    assert "faithfulness" in report.aggregate


def test_ragas_availability_is_boolean():
    # Must never raise, regardless of whether the RAGAS stack imports cleanly.
    assert isinstance(ragas_available(), bool)


def test_ragas_not_configured_without_key():
    cfg = AppConfig.load().with_overrides({"generation": {"openai": {"api_key": ""}}})
    assert RagasEvaluator(cfg).is_configured() is False
