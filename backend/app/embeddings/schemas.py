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
    metadata: dict[str, Any] # The metadata field is a dictionary that can hold any additional information related to the chunk.
    embedding: list[float] # The embedding field is a list of floating-point numbers representing the vector embedding of the chunk's content
    embedding_model: str # The embedding_model field is a string that indicates which model was used to generate the embedding.