"""
Custom RAG evaluation metrics — implemented without RAGAS due to
dependency conflicts with langchain>=0.3.

Metrics implemented:
- Faithfulness: are claims in the answer supported by the retrieved context?
- Answer Relevance: does the answer address the question? (embedding similarity)
- Context Recall: does the retrieved context contain the ground truth information?
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import cast

from backend.app.evaluation.schemas import (
    EvaluationReport,
    EvaluationSample,
    MetricScore,
)

DEFAULT_THRESHOLDS = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "context_recall": 0.6,
}

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return cast(
        np.ndarray,
        model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
    )


def _score_faithfulness(
    model: SentenceTransformer,
    answer: str,
    contexts: list[str],
) -> float:
    """
    Faithfulness: measures whether the answer is grounded in the context.

    Approach: split the answer into sentences, embed each sentence and
    each context chunk, then measure how well each answer sentence is
    supported by the best-matching context chunk. Average across sentences.

    Score of 1.0 = every answer sentence is fully supported by context.
    Score of 0.0 = answer shares no semantic content with context.
    """
    # Split answer into sentences (simple heuristic)
    sentences = [s.strip() for s in answer.replace("?", ".").split(".") if s.strip()]
    if not sentences:
        return 0.0

    context_text = " ".join(contexts)
    all_texts = sentences + [context_text]
    embeddings = _embed(model, all_texts)

    sentence_embeddings = embeddings[:len(sentences)]
    context_embedding = embeddings[len(sentences)]

    scores = [
        _cosine_similarity(sent_emb, context_embedding)
        for sent_emb in sentence_embeddings
    ]
    return float(np.mean(scores))


def _score_answer_relevancy(
    model: SentenceTransformer,
    question: str,
    answer: str,
) -> float:
    """
    Answer Relevance: measures whether the answer addresses the question.

    Approach: embed both question and answer, compute cosine similarity.
    High similarity = answer is semantically relevant to the question.
    """
    embeddings = _embed(model, [question, answer])
    return _cosine_similarity(embeddings[0], embeddings[1])


def _score_context_recall(
    model: SentenceTransformer,
    ground_truth: str,
    contexts: list[str],
) -> float:
    """
    Context Recall: measures whether the retrieved context contains
    the information needed to answer the question (as represented by
    the ground truth answer).

    Approach: embed ground truth and each context chunk, take the
    maximum similarity across all context chunks.
    Score of 1.0 = context perfectly covers the ground truth.
    """
    if not contexts:
        return 0.0

    all_texts = [ground_truth] + contexts
    embeddings = _embed(model, all_texts)

    ground_truth_embedding = embeddings[0]
    context_embeddings = embeddings[1:]

    similarities = [
        _cosine_similarity(ground_truth_embedding, ctx_emb)
        for ctx_emb in context_embeddings
    ]
    return float(max(similarities))


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using embedding-based metrics.
    No external evaluation API required — runs fully locally.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        thresholds: dict = DEFAULT_THRESHOLDS,
    ):
        self.thresholds = thresholds
        self.model = SentenceTransformer(embedding_model)

    def evaluate(self, samples: list[EvaluationSample]) -> EvaluationReport:
        """
        Evaluates a list of samples and returns a structured report.
        """
        if not samples:
            raise ValueError("Cannot evaluate an empty sample list.")

        faithfulness_scores = []
        relevancy_scores = []
        recall_scores = []
        sample_scores = []

        for sample in samples:
            f = _score_faithfulness(
                self.model, sample.generated_answer, sample.retrieved_contexts
            )
            r = _score_answer_relevancy(
                self.model, sample.question, sample.generated_answer
            )
            c = _score_context_recall(
                self.model, sample.ground_truth, sample.retrieved_contexts
            )

            faithfulness_scores.append(f)
            relevancy_scores.append(r)
            recall_scores.append(c)
            sample_scores.append({
                "question": sample.question,
                "faithfulness": round(f, 4),
                "answer_relevancy": round(r, 4),
                "context_recall": round(c, 4),
            })

        metric_results = [
            ("faithfulness", float(np.mean(faithfulness_scores))),
            ("answer_relevancy", float(np.mean(relevancy_scores))),
            ("context_recall", float(np.mean(recall_scores))),
        ]

        metrics = [
            MetricScore(
                name=name,
                score=round(score, 4),
                passed=score >= self.thresholds.get(name, 0.7),
                threshold=self.thresholds.get(name, 0.7),
            )
            for name, score in metric_results
        ]

        return EvaluationReport(
            num_samples=len(samples),
            metrics=metrics,
            sample_scores=sample_scores,
            passed_overall=all(m.passed for m in metrics),
        )