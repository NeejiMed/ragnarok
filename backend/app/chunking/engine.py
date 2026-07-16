import uuid  # importing the uuid module to generate unique identifiers
from typing import Any

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from backend.app.chunking.schemas import ChunkingConfig, DocumentChunk


def chunk_documents(
    documents: list[dict], document_id: str, config: ChunkingConfig, source: str = ""
) -> list[DocumentChunk]:
    """
    Takes a list of extracted document dicts and returns chunks.
    Dispatches to the correct strategy based on config.strategy.
    """
    if config.strategy == "fixed":
        return _fixed_chunking(documents, document_id, config, source)
    elif config.strategy == "recursive":
        return _recursive_chunking(documents, document_id, config, source)
    elif config.strategy == "semantic":
        return _semantic_chunking(documents, document_id, config, source)
    raise ValueError(f"Unknown chunking strategy: {config.strategy}")


def _build_chunk(
    text: str, index: int, document_id: str, strategy: str, base_metadata: dict[str, Any]
) -> DocumentChunk:
    """
    Helper function to build a DocumentChunk object from a text string.
    """
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        content=text.strip(),
        chunk_index=index,
        strategy=strategy,
        metadata={**base_metadata, "chunk_size": len(text)},
    )


def _fixed_chunking(
    documents: list[dict], document_id: str, config: ChunkingConfig, source: str
) -> list[DocumentChunk]:
    """
    Fixed chunking strategy: splits text into fixed-size chunks with optional overlap.
    """
    splitter = CharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, separator="\n"
    )
    chunks = []
    chunk_index = 0
    for doc in documents:
        base_metadata = {
            "page": doc.get("page"),
            "source": source,
            "extraction_method": doc.get("extraction_method"),
        }
        texts = splitter.split_text(doc["content"])
        for text in texts:
            chunks.append(_build_chunk(text, chunk_index, document_id, "fixed", base_metadata))
            chunk_index += 1

    return chunks


def _recursive_chunking(
    documents: list[dict], document_id: str, config: ChunkingConfig, source: str
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = []
    chunk_index = 0
    for doc in documents:
        base_metadata = {
            "page": doc.get("page"),
            "source": source,
            "extraction_method": doc.get("extraction_method"),
        }
        texts = splitter.split_text(doc["content"])
        for text in texts:
            if text.strip():  # Only add non-empty chunks
                chunks.append(
                    _build_chunk(text, chunk_index, document_id, "recursive", base_metadata)
                )
                chunk_index += 1

    return chunks


def _semantic_chunking(
    documents: list[dict], document_id: str, config: ChunkingConfig, source: str
) -> list[DocumentChunk]:
    """
    Semantic chunking strategy using embedding similarity to detect topic shifts.
    Slower than fixed or recursive chunking, but can produce more semantically coherent chunks.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    splitter = SemanticChunker(embeddings)

    chunks = []
    chunk_index = 0
    for doc in documents:
        base_metadata = {
            "page": doc.get("page"),
            "source": source,
            "extraction_method": doc.get("extraction_method"),
        }
        texts = splitter.split_text(doc["content"])
        for text in texts:
            if text.strip():  # Only add non-empty chunks
                chunks.append(
                    _build_chunk(text, chunk_index, document_id, "semantic", base_metadata)
                )
                chunk_index += 1

    return chunks
