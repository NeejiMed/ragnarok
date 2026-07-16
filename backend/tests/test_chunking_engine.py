import pytest

from backend.app.chunking.engine import chunk_documents
from backend.app.chunking.schemas import ChunkingConfig


def make_long_doc(
    num_pages: int = 1,
    page_length: int = 300,
    content_prefix: str = "Page content",
) -> list[dict]:
    """
    Helper function to create a long document with multiple pages.
    Each page is represented as a dictionary with 'page' and 'content' keys.
    """
    return [
        {"page": i + 1, "content": f"{content_prefix} {i + 1}: " + "word " * page_length}
        for i in range(num_pages)
    ]


# Fixed strategy on a long document with multiple pages
def test_fixed_chunking():
    documents = make_long_doc(num_pages=3, page_length=300)
    config = ChunkingConfig(strategy="fixed", chunk_size=500, chunk_overlap=50)
    document_id = "doc1"
    source = "test_source"

    chunks = chunk_documents(documents, document_id, config, source)

    # Check that we have multiple chunks
    assert len(chunks) > 1

    # Check that all chunks have the correct strategy
    for chunk in chunks:
        assert chunk.strategy == "fixed"
        assert chunk.document_id == document_id
        assert chunk.metadata["source"] == source

    # Check that chunk_index increments correctly
    for i in range(len(chunks)):
        assert chunks[i].chunk_index == i


def test_recursive_chunking():
    documents = make_long_doc(num_pages=2, page_length=400)
    config = ChunkingConfig(strategy="recursive", chunk_size=500, chunk_overlap=50)
    document_id = "doc2"
    source = "test_source"
    chunks = chunk_documents(documents, document_id, config, source)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.strategy == "recursive"
        assert chunk.document_id == document_id
        assert chunk.metadata["source"] == source

    for i in range(len(chunks)):
        assert chunks[i].chunk_index == i


# test for invalid strategy
def test_invalid_strategy():
    documents = make_long_doc(num_pages=1, page_length=300)
    config = ChunkingConfig(strategy="invalid_strategy", chunk_size=500, chunk_overlap=50)
    document_id = "doc3"
    source = "test_source"

    with pytest.raises(ValueError):
        chunk_documents(documents, document_id, config, source)


# chunk metadata inheritance -> page from parent document should appear in chunk metadata
def test_chunk_metadata_inheritance():
    documents = make_long_doc(num_pages=1, page_length=300)
    config = ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=10)
    document_id = "doc4"
    source = "test_source"

    chunks = chunk_documents(documents, document_id, config, source)

    for chunk in chunks:
        assert chunk.metadata["source"] == source
        assert (
            chunk.metadata["page"] == 1
        )  # All chunks should inherit the page number from the parent document


# test for empty document list
def test_empty_document_list():
    documents = []
    config = ChunkingConfig(strategy="fixed", chunk_size=500, chunk_overlap=50)
    document_id = "doc5"
    source = "test_source"

    chunks = chunk_documents(documents, document_id, config, source)

    assert len(chunks) == 0
    assert chunks == []  # Should return an empty list when no documents are provided
    assert isinstance(chunks, list)  # Ensure the return type is a list
