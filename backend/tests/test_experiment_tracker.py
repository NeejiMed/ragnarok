from unittest.mock import MagicMock, patch

from backend.app.evaluation.schemas import EvaluationReport, MetricScore
from backend.app.evaluation.tracker import (
    ExperimentTracker,
    run_chunking_experiment,
    run_embedding_experiment,
)


def make_report(passed: bool = True) -> EvaluationReport:
    return EvaluationReport(
        num_samples=2,
        metrics=[
            MetricScore(name="faithfulness", score=0.82, passed=True, threshold=0.7),
            MetricScore(name="answer_relevancy", score=0.75, passed=True, threshold=0.7),
            MetricScore(name="context_recall", score=0.68, passed=passed, threshold=0.6),
        ],
        sample_scores=[],
        passed_overall=passed,
    )


#  Pure unit tests (mlflow fully mocked)


@patch("backend.app.evaluation.tracker.mlflow")
def test_tracker_sets_tracking_uri_on_init(mock_mlflow):
    """Tracker must configure MLflow's tracking URI from settings."""
    ExperimentTracker("test_experiment")
    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once_with("test_experiment")


@patch("backend.app.evaluation.tracker.mlflow")
def test_log_params_delegates_to_mlflow(mock_mlflow):
    tracker = ExperimentTracker("test")
    tracker.log_params({"strategy": "recursive", "chunk_size": 500})
    mock_mlflow.log_params.assert_called_once_with({"strategy": "recursive", "chunk_size": 500})


@patch("backend.app.evaluation.tracker.mlflow")
def test_log_metrics_delegates_to_mlflow(mock_mlflow):
    tracker = ExperimentTracker("test")
    tracker.log_metrics({"faithfulness": 0.82})
    mock_mlflow.log_metrics.assert_called_once_with({"faithfulness": 0.82}, step=None)


@patch("backend.app.evaluation.tracker.mlflow")
def test_log_evaluation_report_logs_all_metric_scores(mock_mlflow):
    """Every metric in the report must be logged to MLflow."""
    # Mock the context manager for start_run
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    tracker = ExperimentTracker("test")
    report = make_report()
    tracker.log_evaluation_report(report)

    # Check that log_metric was called for each metric + summary metrics
    logged_metric_names = {call_args[0][0] for call_args in mock_mlflow.log_metric.call_args_list}
    assert "faithfulness" in logged_metric_names
    assert "answer_relevancy" in logged_metric_names
    assert "context_recall" in logged_metric_names
    assert "num_samples" in logged_metric_names
    assert "passed_overall" in logged_metric_names


@patch("backend.app.evaluation.tracker.mlflow")
def test_run_chunking_experiment_logs_strategy_params(mock_mlflow):
    """Chunking experiment must log strategy, chunk_size, and chunk_overlap."""
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    tracker = ExperimentTracker("chunking_experiment")
    run_chunking_experiment(
        tracker=tracker,
        strategy="recursive",
        chunk_size=500,
        chunk_overlap=50,
        evaluation_report=make_report(),
    )

    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert logged_params["chunking_strategy"] == "recursive"
    assert logged_params["chunk_size"] == 500
    assert logged_params["chunk_overlap"] == 50


@patch("backend.app.evaluation.tracker.mlflow")
def test_run_embedding_experiment_logs_model_name(mock_mlflow):
    """Embedding experiment must log the model name as a parameter."""
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    tracker = ExperimentTracker("embedding_experiment")
    run_embedding_experiment(
        tracker=tracker,
        model_name="BAAI/bge-small-en-v1.5",
        evaluation_report=make_report(),
        retrieval_latency_ms=120.5,
    )

    # Verify model name logged as parameter
    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert logged_params["embedding_model"] == "BAAI/bge-small-en-v1.5"

    # Verify latency logged as metric — check all log_metrics calls
    all_metric_dicts = [
        call_args[0][0] for call_args in mock_mlflow.log_metrics.call_args_list if call_args[0]
    ]
    latency_logged = any("retrieval_latency_ms" in metrics for metrics in all_metric_dicts)
    assert latency_logged, (
        f"Expected retrieval_latency_ms in logged metrics. Got: {all_metric_dicts}"
    )
