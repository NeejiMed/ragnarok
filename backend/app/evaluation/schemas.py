from pydantic import BaseModel

class EvaluationSample(BaseModel):
    """One question-answer pair with its retrieved context and ground truth."""
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: list[str] # list of chunk content strings


class MetricScore(BaseModel):
    """Score for one metric with pass/fail threshold."""
    name: str
    score: float
    passed: bool
    threshold: float

class EvaluationReport(BaseModel):
    """Complete evaluation results across all samples and metrics."""
    num_samples: int
    metrics: list[MetricScore]
    sample_scores: list[dict]  # list of dicts with sample-level scores for each metric
    passed_overall: bool       # True if all metrics passed their thresholds, False otherwise
    