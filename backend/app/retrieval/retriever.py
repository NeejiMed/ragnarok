from backend.app.chunking.schemas import DocumentChunk
from backend.app.embeddings.pipeline import EmbeddingPipeline
from backend.app.vectorstore.store import VectorStore


class Retriever:
    """
    Layer 1: Basic semantic retrieval.
    Embeds a query and searches Qdrant for the most similar chunks.
    """

    def __init__(self, vector_store: VectorStore, embedding_pipeline: EmbeddingPipeline):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        """
        Embeds the query and retrieves the top-K relevant chunks from Qdrant.
        Returns a list of playload dicts with similarity score.
        """
        # Embed the query as a single item list (pipeline expects a list of strings
        query_chunk = DocumentChunk(
            chunk_id="query",
            document_id="query",
            content=query,
            chunk_index=0,
            strategy="query",
            metadata={},
        )
        embedded = self.embedding_pipeline.embed([query_chunk])
        query_vector = embedded[0].embedding

        return self.vector_store.search(query_vector=query_vector, top_k=top_k, filters=filters)
