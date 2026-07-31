from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.app.core.config import settings
from backend.app.embeddings.schemas import EmbeddedChunk


class VectorStore:
    """
    Manages persistence and retrieval of EmbeddedChunks in Qdrant.
    One instance per application — holds the Qdrant client connection.
    """

    def __init__(
        self,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        collection_name: str = settings.qdrant_collection,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def ensure_collection(self, embedding_dimension: int) -> None:
        """
        Creates the Qdrant collection if it doesn't exist.
        Safe to call on every startup — idempotent.
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        """
        Inserts or updates EmbeddedChunks in Qdrant.
        Returns the number of chunks upserted.
        Upsert (not insert) so re-processing a document updates existing vectors
        rather than creating duplicates.
        """
        if not chunks:
            return 0

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "strategy": chunk.strategy,
                    "embedding_model": chunk.embedding_model,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Searches for the top-K most similar vectors.
        Optional filters: e.g. {"document_id": "abc123"} to search within one doc.
        Returns list of payload dicts with similarity score added.
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                **(result.payload or {}),
                "score": result.score,
                "chunk_id": str(result.id),
            }
            for result in results.points
        ]

    def delete_by_document_id(self, document_id: str) -> None:
        """
        Deletes all chunks belonging to a document.
        Used when a document is re-uploaded — clean slate before re-embedding.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

    def _build_filter(self, filters: dict) -> Filter:
        """Converts a plain dict of filters into a Qdrant Filter object."""
        conditions: list[Condition] = [
            FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()
        ]
        return Filter(must=conditions)
