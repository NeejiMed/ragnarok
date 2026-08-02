from uuid import uuid4

import pytest

from backend.app.chunking.schemas import DocumentChunk
from backend.app.retrieval.retriever import Retriever
from backend.app.vectorstore.store import VectorStore


@pytest.fixture
def vector_store(in_memory_qdrant_client):
    """Fresh isolated Qdrant collection per test, deleted after."""
    collection_name = f"test_{uuid4().hex[:8]}"
    store = VectorStore(
        collection_name=collection_name,
        client=in_memory_qdrant_client,
    )
    store.ensure_collection(embedding_dimension=384)
    yield store
    store.client.delete_collection(collection_name)


@pytest.fixture
def seeded_vector_store(vector_store, embedding_pipeline):
    """Vector store pre-populated with three domain-realistic test chunks."""
    test_contents = [
        ("The company refund policy allows returns within 30 days.", "doc1"),
        ("Employees are entitled to 20 days of annual leave.", "doc2"),
        ("The IT helpdesk can be reached at helpdesk@company.com.", "doc3"),
    ]
    chunks_to_embed = [
        DocumentChunk(
            chunk_id=str(uuid4()),
            document_id=doc_id,
            content=content,
            chunk_index=i,
            strategy="test",
            metadata={"source": "test_doc"},
        )
        for i, (content, doc_id) in enumerate(test_contents)
    ]
    embedded = embedding_pipeline.embed(chunks_to_embed)
    vector_store.upsert(embedded)
    return vector_store


@pytest.mark.slow
def test_retrieve_returns_results(seeded_vector_store, embedding_pipeline):
    retriever = Retriever(seeded_vector_store, embedding_pipeline)
    results = retriever.retrieve("What is the refund policy?", top_k=3)
    assert len(results) > 0


@pytest.mark.slow
def test_retrieve_results_have_expected_fields(seeded_vector_store, embedding_pipeline):
    retriever = Retriever(seeded_vector_store, embedding_pipeline)
    results = retriever.retrieve("annual leave entitlement", top_k=2)
    for result in results:
        assert "content" in result, "Missing content  RAG pipeline will fail"
        assert "score" in result, "Missing score  reranker will fail"
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0


@pytest.mark.slow
def test_retrieve_respects_top_k(seeded_vector_store, embedding_pipeline):
    retriever = Retriever(seeded_vector_store, embedding_pipeline)
    results = retriever.retrieve("company policy", top_k=1)
    assert len(results) == 1


@pytest.mark.slow
def test_retrieve_with_filter_returns_only_matching_documents(
    seeded_vector_store, embedding_pipeline
):
    retriever = Retriever(seeded_vector_store, embedding_pipeline)
    results = retriever.retrieve(
        "company information",
        top_k=5,
        filters={"document_id": "doc1"},
    )
    assert len(results) > 0
    for result in results:
        assert (
            result["document_id"] == "doc1"
        ), f"Filter failed  got document_id={result['document_id']}, expected doc1"


@pytest.mark.slow
def test_retrieve_most_relevant_result_is_top_ranked(seeded_vector_store, embedding_pipeline):
    retriever = Retriever(seeded_vector_store, embedding_pipeline)
    results = retriever.retrieve("What is the refund policy?", top_k=3)
    assert len(results) >= 1
    top_result = results[0]
    assert (
        "refund" in top_result["content"].lower()
    ), f"Expected refund-related content at top, got: '{top_result['content']}'"
