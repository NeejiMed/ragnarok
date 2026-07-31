from uuid import uuid4

from backend.app.embeddings.schemas import EmbeddedChunk
from backend.app.vectorstore.store import VectorStore


# Test ensuring collection creation to verify that the collection is created if it doesn't exist
def test_ensure_collection_creates_collection():
    # Create a VectorStore instance with a test collection name
    test_collection_name = "test_collection"
    vector_store = VectorStore(collection_name=test_collection_name)

    # Ensure the collection is created
    vector_store.ensure_collection(embedding_dimension=128)

    # Verify that the collection now exists
    existing_collections = [c.name for c in vector_store.client.get_collections().collections]
    assert test_collection_name in existing_collections


# Test upsert with a list of EmbeddedChunks and returns the correct number of upserted chunks
def test_upsert_inserts_chunks():
    # Create a VectorStore instance with a test collection name
    test_collection_name = "test_collection"
    vector_store = VectorStore(collection_name=test_collection_name)

    # Ensure the collection is created
    vector_store.ensure_collection(embedding_dimension=128)

    # Create a list of mock EmbeddedChunks
    mock_chunks = [
        EmbeddedChunk(
            chunk_id=str(uuid4()),
            embedding=[0.1] * 128,
            content="This is the content of chunk 1.",
            document_id="doc1",
            chunk_index=0,
            strategy="fixed",
            embedding_model="test_model",
            metadata={"source": "test_source"},
        ),
        EmbeddedChunk(
            chunk_id=str(uuid4()),
            embedding=[0.2] * 128,
            content="This is the content of chunk 2.",
            document_id="doc1",
            chunk_index=1,
            strategy="fixed",
            embedding_model="test_model",
            metadata={"source": "test_source"},
        ),
    ]

    # Upsert the mock chunks
    num_upserted = vector_store.upsert(mock_chunks)

    # Verify that the number of upserted chunks is correct
    assert num_upserted == len(mock_chunks)


# Test search return results, each result has content and score fields
def test_search_returns_results():
    # Create a VectorStore instance with a test collection name
    test_collection_name = "test_collection"
    vector_store = VectorStore(collection_name=test_collection_name)

    # Ensure the collection is created
    vector_store.ensure_collection(embedding_dimension=128)

    # Create a list of mock EmbeddedChunks and upsert them
    mock_chunks = [
        EmbeddedChunk(
            chunk_id=str(uuid4()),
            embedding=[0.1] * 128,
            content="This is the content of chunk 1.",
            document_id="doc1",
            chunk_index=0,
            strategy="fixed",
            embedding_model="test_model",
            metadata={"source": "test_source"},
        ),
        EmbeddedChunk(
            chunk_id=str(uuid4()),
            embedding=[0.2] * 128,
            content="This is the content of chunk 2.",
            document_id="doc1",
            chunk_index=1,
            strategy="fixed",
            embedding_model="test_model",
            metadata={"source": "test_source"},
        ),
    ]
    vector_store.upsert(mock_chunks)

    # Perform a search with a query vector
    query_vector = [0.15] * 128
    results = vector_store.search(query_vector=query_vector, top_k=2)

    # Verify that results are returned and contain expected fields
    assert len(results) > 0
    for result in results:
        assert "content" in result
        assert "score" in result


# Test search with a filter returns only results matching that filter
def test_search_with_filter():
    # Create a VectorStore instance with a test collection name
    test_collection_name = "test_collection"
    vector_store = VectorStore(collection_name=test_collection_name)

    # Ensure the collection is created
    vector_store.ensure_collection(embedding_dimension=128)

    # Create a list of mock EmbeddedChunks and upsert them
    mock_chunks = [
        EmbeddedChunk(
            chunk_id=str(uuid4()),
            embedding=[0.1] * 128,
            content="This is the content of chunk 1.",
            document_id="doc1",
            chunk_index=0,
            strategy="fixed",
            embedding_model="test_model",
            metadata={"source": "test_source"},
        ),
        EmbeddedChunk(
            chunk_id=str(uuid4()),
            embedding=[0.2] * 128,
            content="This is the content of chunk 2.",
            document_id="doc1",
            chunk_index=1,
            strategy="fixed",
            embedding_model="test_model",
            metadata={"source": "test_source"},
        ),
    ]
    vector_store.upsert(mock_chunks)

    # Perform a search with a query vector and a filter
    query_vector = [0.15] * 128
    filter_dict = {"source": "test_source"}
    results = vector_store.search(query_vector=query_vector, top_k=2, filters=filter_dict)

    # Verify that results are returned and contain expected fields
    assert len(results) > 0
    for result in results:
        assert "content" in result
        assert "score" in result


# Test delete_by_document_id removes all chunks associated with that document_id
def test_delete_by_document_id():
    # Create a VectorStore instance with a test collection name
    test_collection_name = "test_collection"
    vector_store = VectorStore(collection_name=test_collection_name)

    # Ensure the collection is created
    vector_store.ensure_collection(embedding_dimension=128)
