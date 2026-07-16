from typing import Any

from pydantic import BaseModel


class ChunkingConfig(BaseModel):
    """Configuration for chunking behavior."""

    strategy: str = "recursive"  # Strategy for chunking, e.g., "fixed", "recursive", "semantic"
    chunk_size: int = 500  # Size of each chunk in characters
    chunk_overlap: int = 50  # Overlap between chunks in characters


class DocumentChunk(BaseModel):
    """A single chunk produced by the chunking engine."""

    chunk_id: str  # Unique identifier for the chunk
    document_id: str  # Identifier for the original document
    content: str  # The actual text content of the chunk
    chunk_index: int  # Index of the chunk in the sequence of chunks
    strategy: str  # The chunking strategy used to produce this chunk
    metadata: dict[
        str, Any
    ]  # Additional metadata associated with the chunk, such as source, page number, etc.
