import json
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import params
import mlflow

from backend.app.core.config import settings
from backend.app.evaluation.schemas import EvaluationReport

class ExperimentTracker:
    """
    Wraps MLflow to track RAG experiments.
    Provides a clean interface so the rest of the codebase doesn't
    import mlflow directly, keeps MLflow swappable
    """

    def __init__(self, experiment_name: str):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    @contextmanager
    def start_run(self, run_name:str, tags: dict | None = None):
        """
        Context manager for a single experiment run. 
        Automatically ends thee run on exit, even if an exception occurs.
        Usage:
            with tracker.start_run("recursive_chunking_test") as run:
                tracker.log_params({"strategy": "recursive"})
                tracker.log_metrics({"accuracy": 0.78})
        """
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run # yield the run object so the caller can access it if needed

    def log_params(self, params: dict) -> None:
        """
        Log parameters for the current run.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """
        Log metrics for the current run.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_evaluation_report(self, report: EvaluationReport) -> None:
        """
        Logs all metrics from an EvaluationReport and saves the full
        report as a JSON artifact for later inspection.
        """
        # log aggregated metrics scores
        for metric in report.metrics:
            mlflow.log_metric(metric.name, metric.score)

        mlflow.log_metric("num_samples", report.num_samples)
        mlflow.log_metric("passed_overall", float(report.passed_overall))

        # save full report as JSON artifact
        report_path = Path("evaluation_report.json")
        report_path.write_text(report.model_dump_json(indent=2))
        mlflow.log_artifact(str(report_path))
        report_path.unlink()  # clean up the temporary file

    def log_artifact(self, file_path: str) -> None:
        """Logs any file as an artifact for the current run."""
        mlflow.log_artifact(file_path)

def run_chunking_experiment(
    tracker: ExperimentTracker,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    evaluation_report: EvaluationReport,
    extra_params: dict | None = None,
) -> None:
    """
    Convenience function for logging a chunking strategy experiment.
    Encapsulates the standard parameter + metric logging pattern
    so callers don't need to know MLflow's API directly.
    """
    params = {
        "chunking_strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        **(extra_params or {}),
    }

    with tracker.start_run(run_name=f"{strategy}_chunk{chunk_size}"):
        tracker.log_params(params)
        tracker.log_evaluation_report(evaluation_report)

def run_embedding_experiment(
    tracker: ExperimentTracker,
    model_name: str,
    evaluation_report: EvaluationReport,
    retrieval_latency_ms: float | None = None,
    extra_params: dict | None = None,
) -> None:
    """
    Convenience function for logging an embedding model experiment.
    """
    params = {
        "embedding_model": model_name,
        **(extra_params or {}),
    }

    with tracker.start_run(run_name=f"embedding_{model_name.replace('/', '_')}"):
        tracker.log_params(params)
        tracker.log_evaluation_report(evaluation_report)
        if retrieval_latency_ms is not None:
            tracker.log_metrics({"retrieval_latency_ms": retrieval_latency_ms})