from uuid import uuid4

import numpy as np
import pytest

from backend.app.embeddings.schemas import EmbeddedChunk
from backend.app.vectorstore.store import VectorStore


def make_test_chunk(content: str, document_id: str, index: int) -> EmbeddedChunk:
    """Creates a normalized test EmbeddedChunk without needing the real embedding model."""
    vector = np.random.rand(128)
    vector = (vector / np.linalg.norm(vector)).tolist()
    return EmbeddedChunk(
        chunk_id=str(uuid4()),
        embedding=vector,
        content=content,
        document_id=document_id,
        chunk_index=index,
        strategy="fixed",
        embedding_model="test_model",
        metadata={"source": "test_source"},
    )


@pytest.fixture
def vector_store(in_memory_qdrant_client):
    """Creates a fresh isolated collection per test, deletes it after."""
    collection_name = f"test_{uuid4().hex[:8]}"
    store = VectorStore(
        collection_name=collection_name,
        client=in_memory_qdrant_client,
    )
    store.ensure_collection(embedding_dimension=128)
    yield store
    store.client.delete_collection(collection_name)


def test_ensure_collection_creates_collection(vector_store):
    existing = [c.name for c in vector_store.client.get_collections().collections]
    assert vector_store.collection_name in existing


def test_ensure_collection_is_idempotent(vector_store):
    """Calling ensure_collection twice must not raise."""
    vector_store.ensure_collection(embedding_dimension=128)
    existing = [c.name for c in vector_store.client.get_collections().collections]
    assert vector_store.collection_name in existing


def test_upsert_returns_correct_count(vector_store):
    chunks = [
        make_test_chunk("Content of chunk 1.", "doc1", 0),
        make_test_chunk("Content of chunk 2.", "doc1", 1),
    ]
    assert vector_store.upsert(chunks) == 2


def test_search_returns_results_with_expected_fields(vector_store):
    chunks = [
        make_test_chunk("Content of chunk 1.", "doc1", 0),
        make_test_chunk("Content of chunk 2.", "doc1", 1),
    ]
    vector_store.upsert(chunks)

    query_vector = (np.ones(128) / np.linalg.norm(np.ones(128))).tolist()
    results = vector_store.search(query_vector=query_vector, top_k=2)

    assert len(results) > 0
    for result in results:
        assert "content" in result
        assert "score" in result


def test_search_with_filter_returns_matching_results(vector_store):
    chunks = [
        make_test_chunk("Doc A content.", "doc_a", 0),
        make_test_chunk("Doc B content.", "doc_b", 0),
    ]
    vector_store.upsert(chunks)

    query_vector = (np.ones(128) / np.linalg.norm(np.ones(128))).tolist()
    results = vector_store.search(
        query_vector=query_vector,
        top_k=5,
        filters={"document_id": "doc_a"},
    )

    assert len(results) > 0
    for result in results:
        assert result["document_id"] == "doc_a"


def test_delete_by_document_id_removes_chunks(vector_store):
    chunks = [
        make_test_chunk("Content 1.", "doc1", 0),
        make_test_chunk("Content 2.", "doc1", 1),
    ]
    vector_store.upsert(chunks)
    vector_store.delete_by_document_id("doc1")

    query_vector = (np.ones(128) / np.linalg.norm(np.ones(128))).tolist()
    results = vector_store.search(
        query_vector=query_vector,
        top_k=5,
        filters={"document_id": "doc1"},
    )
    assert len(results) == 0
