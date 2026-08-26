from unittest.mock import patch

import numpy as np
import pytest

from backend.app.evaluation.evaluator import (
    DEFAULT_THRESHOLDS,
    RAGEvaluator,
    _cosine_similarity,
    _score_answer_relevancy,
    _score_context_recall,
    _score_faithfulness,
)
from backend.app.evaluation.schemas import (
    EvaluationReport,
    EvaluationSample,
    MetricScore,
)

#  Helpers


def make_sample(
    question: str = "What is the refund policy?",
    ground_truth: str = "Returns are allowed within 30 days.",
    generated_answer: str = "The refund policy allows returns within 30 days.",
    retrieved_contexts: list[str] | None = None,
) -> EvaluationSample:
    return EvaluationSample(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer,
        retrieved_contexts=retrieved_contexts
        or ["The company refund policy allows returns within 30 days of purchase."],
    )


#  Pure unit tests: schemas


def test_evaluation_sample_schema_accepts_valid_input():
    sample = make_sample()
    assert sample.question == "What is the refund policy?"
    assert len(sample.retrieved_contexts) == 1


def test_metric_score_pass_when_above_threshold():
    metric = MetricScore(name="faithfulness", score=0.85, passed=True, threshold=0.7)
    assert metric.passed is True


def test_metric_score_fail_when_below_threshold():
    metric = MetricScore(name="faithfulness", score=0.55, passed=False, threshold=0.7)
    assert metric.passed is False


def test_evaluation_report_passed_overall_requires_all_metrics_pass():
    """passed_overall must be False if any single metric fails."""
    report = EvaluationReport(
        num_samples=1,
        metrics=[
            MetricScore(name="faithfulness", score=0.9, passed=True, threshold=0.7),
            MetricScore(name="answer_relevancy", score=0.4, passed=False, threshold=0.7),
            MetricScore(name="context_recall", score=0.8, passed=True, threshold=0.6),
        ],
        sample_scores=[],
        passed_overall=False,
    )
    assert report.passed_overall is False


def test_default_thresholds_are_reasonable():
    """Thresholds must be between 0 and 1 and cover all three metrics."""
    required = {"faithfulness", "answer_relevancy", "context_recall"}
    assert required == set(DEFAULT_THRESHOLDS.keys())
    for name, threshold in DEFAULT_THRESHOLDS.items():
        assert 0.0 < threshold < 1.0, (
            f"Threshold for {name} must be between 0 and 1, got {threshold}"
        )


def test_cosine_similarity_identical_vectors():
    """Identical vectors must have cosine similarity of 1.0."""
    v = np.array([0.1, 0.5, 0.3, 0.8])
    v = v / np.linalg.norm(v)
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors must have cosine similarity of 0.0."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_evaluator_raises_on_empty_samples():
    """Empty sample list must raise ValueError."""
    with patch("backend.app.evaluation.evaluator.SentenceTransformer"):
        evaluator = RAGEvaluator()
    with pytest.raises(ValueError, match="Cannot evaluate an empty sample list"):
        evaluator.evaluate([])


#  Slow tests: real embedding model


@pytest.mark.slow
def test_faithfulness_high_for_grounded_answer(embedding_pipeline):
    """
    An answer that closely mirrors context content should score high
    on faithfulness — the answer is grounded in the retrieved context.
    """
    score = _score_faithfulness(
        embedding_pipeline.model,
        answer="Returns are allowed within 30 days of purchase.",
        contexts=["The company refund policy allows returns within 30 days of purchase."],
    )
    assert score > 0.7, f"Expected faithfulness > 0.7, got {score:.3f}"


@pytest.mark.slow
def test_faithfulness_low_for_ungrounded_answer(embedding_pipeline):
    """
    An answer about a completely unrelated topic should score low
    on faithfulness — not supported by the retrieved context.
    """
    score = _score_faithfulness(
        embedding_pipeline.model,
        answer="The weather today is sunny and warm.",
        contexts=["The company refund policy allows returns within 30 days."],
    )
    assert score < 0.7, f"Expected faithfulness < 0.7, got {score:.3f}"


@pytest.mark.slow
def test_answer_relevancy_high_for_on_topic_answer(embedding_pipeline):
    """Answer about refund policy should be highly relevant to a refund question."""
    score = _score_answer_relevancy(
        embedding_pipeline.model,
        question="What is the refund policy?",
        answer="You can return items within 30 days for a full refund.",
    )
    assert score > 0.7, f"Expected relevancy > 0.7, got {score:.3f}"


@pytest.mark.slow
def test_context_recall_high_when_context_covers_ground_truth(embedding_pipeline):
    """Context that contains the ground truth answer should score high recall."""
    score = _score_context_recall(
        embedding_pipeline.model,
        ground_truth="Returns are allowed within 30 days.",
        contexts=["The refund policy allows product returns within 30 days of purchase."],
    )
    assert score > 0.7, f"Expected recall > 0.7, got {score:.3f}"


@pytest.mark.slow
def test_full_evaluator_returns_report_with_all_metrics(embedding_pipeline):
    """
    End-to-end evaluator test: must return report with all three metrics
    and correct sample count.
    """
    evaluator = RAGEvaluator()
    evaluator.model = embedding_pipeline.model  # reuse already-loaded model

    samples = [
        make_sample(),
        make_sample(
            question="How many leave days do employees get?",
            ground_truth="Employees get 20 days of annual leave.",
            generated_answer="Employees are entitled to 20 days of annual leave per year.",
            retrieved_contexts=["Employees are entitled to 20 days of annual leave."],
        ),
    ]

    report = evaluator.evaluate(samples)

    assert report.num_samples == 2
    assert len(report.metrics) == 3
    metric_names = {m.name for m in report.metrics}
    assert metric_names == {"faithfulness", "answer_relevancy", "context_recall"}
    assert len(report.sample_scores) == 2
    for score_dict in report.sample_scores:
        assert "faithfulness" in score_dict
        assert "answer_relevancy" in score_dict
        assert "context_recall" in score_dict
