from typing import Any

from pydantic import BaseModel


class EmbeddedChunk(BaseModel):
    """
    A documentChunk enriched with its embedding vector.
    """

    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    strategy: str
    metadata: dict[str, Any]
    embedding: list[float]
    embedding_model: str
