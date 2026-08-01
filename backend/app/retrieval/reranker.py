from sentence_transformers import CrossEncoder

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    """
    Layer 3: Re-scores retrieved chunks using a cross-encoder model to improve relevance.
    Cross-encoders read query+chunk together, more accurate than bi-encoders cosine similarity
    but too slow to run on the full corpus.
    Use AFTER initial retrieval to re-rank a small candidate set of chunks (e.g. top 20).
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
            self,
            query: str,
            results: list[dict],
            top_k: int = 5
    ) -> list[dict]:
        """
        Re-scores results using cross-encoder model and returns top-K most relevant chunks.
        """

        if not results:
            return []

        # Prepare pairs of (query, chunk content) for scoring
        pairs = [(query, result["content"]) for result in results]
        scores = self.model.predict(pairs)

        # Attach scores to results
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        # Sort by rerank_score descending and return top-K
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]