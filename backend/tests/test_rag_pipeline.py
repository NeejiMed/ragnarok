from uuid import uuid4
from unittest.mock import MagicMock, patch

import pytest

from backend.app.chunking.schemas import DocumentChunk
from backend.app.embeddings.pipeline import EmbeddingPipeline
from backend.app.rag.pipeline import RAGPipeline
from backend.app.rag.prompt import format_context
from backend.app.rag.schemas import RAGRequest
from backend.app.vectorstore.store import VectorStore


@pytest.fixture
def vector_store(in_memory_qdrant_client):
    """
    Fresh isolated collection per test using the in-memory client.
    No Docker required  qdrant-client handles storage in process memory.
    Collection is deleted after each test to prevent cross-test pollution.
    """
    collection_name = f"test_rag_{uuid4().hex[:8]}"
    store = VectorStore(
        collection_name=collection_name,
        client=in_memory_qdrant_client,
    )
    store.ensure_collection(embedding_dimension=384)
    yield store
    store.client.delete_collection(collection_name)


@pytest.fixture
def seeded_vector_store(vector_store, embedding_pipeline):
    """
    Pre-populates the store with three domain-realistic chunks.
    Uses real embeddings so semantic retrieval assertions are meaningful.
    """
    chunks = [
        DocumentChunk(
            chunk_id=str(uuid4()),
            document_id="refund-policy",
            content="The company refund policy allows returns within 30 days.",
            chunk_index=0,
            strategy="test",
            metadata={"source": "policy-handbook", "page": 3},
        ),
        DocumentChunk(
            chunk_id=str(uuid4()),
            document_id="leave-policy",
            content="Employees are entitled to 20 days of annual leave.",
            chunk_index=0,
            strategy="test",
            metadata={"source": "hr-guide", "page": 8},
        ),
        DocumentChunk(
            chunk_id=str(uuid4()),
            document_id="helpdesk",
            content="The IT helpdesk can be reached at helpdesk@company.com.",
            chunk_index=0,
            strategy="test",
            metadata={"source": "it-faq", "page": 1},
        ),
    ]
    embedded = embedding_pipeline.embed(chunks)
    vector_store.upsert(embedded)
    return vector_store


# ── Pure unit tests ───────────────────────────────────────────────────────────

def test_format_context_includes_source_citation():
    results = [
        {"source": "policy-handbook", "page": 3, "content": "Returns allowed within 30 days."},
        {"source": "hr-guide", "content": "Employees get annual leave."},
    ]
    context = format_context(results)

    assert "[Source 1: policy-handbook, Page 3]" in context
    assert "Returns allowed within 30 days." in context
    assert "[Source 2: hr-guide]" in context
    assert "Employees get annual leave." in context


def test_format_context_no_page_omits_page_field():
    results = [{"source": "wiki", "content": "Some content.", "page": None}]
    context = format_context(results)

    assert "Page None" not in context
    assert "[Source 1: wiki]" in context


# ── Integration + slow tests ──────────────────────────────────────────────────

@pytest.mark.slow
def test_rag_pipeline_returns_correct_response_shape(
    seeded_vector_store, embedding_pipeline
):
    """Full pipeline with mocked LLM  tests orchestration, not LLM quality."""
    with patch("backend.app.rag.pipeline.OllamaLLM") as mock_llm_class, \
         patch("backend.app.rag.pipeline.Reranker") as mock_reranker_class:

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The refund window is 30 days."
        mock_llm_class.return_value = mock_llm
        mock_reranker_class.return_value = MagicMock()

        pipeline = RAGPipeline(seeded_vector_store, embedding_pipeline)
        response = pipeline.run(RAGRequest(question="What is the refund policy?"))

    assert response.question == "What is the refund policy?"
    assert response.answer == "The refund window is 30 days."
    assert response.retrieval_count > 0
    assert response.model_used == "llama3.2"
    assert len(response.sources) == response.retrieval_count
    assert response.sources[0].content != ""
    assert response.sources[0].score >= 0.0
    mock_llm.invoke.assert_called_once()


@pytest.mark.slow
def test_rag_pipeline_llm_receives_context_in_prompt(
    seeded_vector_store, embedding_pipeline
):
    """Retrieved content must actually reach the LLM prompt  not just be retrieved."""
    with patch("backend.app.rag.pipeline.OllamaLLM") as mock_llm_class, \
         patch("backend.app.rag.pipeline.Reranker") as mock_reranker_class:

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Mocked answer."
        mock_llm_class.return_value = mock_llm
        mock_reranker_class.return_value = MagicMock()

        pipeline = RAGPipeline(seeded_vector_store, embedding_pipeline)
        pipeline.run(RAGRequest(question="What is the refund policy?"))

    actual_prompt = mock_llm.invoke.call_args[0][0]
    assert "refund" in actual_prompt.lower(), (
        "Refund policy content must appear in the LLM prompt"
    )
    assert "What is the refund policy?" in actual_prompt


@pytest.mark.slow
def test_rag_pipeline_with_reranking_calls_reranker(
    seeded_vector_store, embedding_pipeline
):
    """use_reranking=True must invoke reranker and use rerank_score in response."""
    reranked_result = {
        "content": "The company refund policy allows returns within 30 days.",
        "score": 0.91,
        "rerank_score": 0.99,
        "source": "policy-handbook",
        "page": 3,
        "document_id": "refund-policy",
    }

    with patch("backend.app.rag.pipeline.OllamaLLM") as mock_llm_class, \
         patch("backend.app.rag.pipeline.Reranker") as mock_reranker_class:

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The refund window is 30 days."
        mock_llm_class.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [reranked_result]
        mock_reranker_class.return_value = mock_reranker

        pipeline = RAGPipeline(seeded_vector_store, embedding_pipeline)
        response = pipeline.run(
            RAGRequest(question="What is the refund policy?", use_reranking=True)
        )

    mock_reranker.rerank.assert_called_once()
    mock_llm.invoke.assert_called_once()
    assert response.retrieval_count == 1
    assert response.sources[0].score == 0.99


@pytest.mark.slow
def test_rag_pipeline_empty_retrieval_sends_fallback_in_prompt(
    seeded_vector_store, embedding_pipeline
):
    """
    Fallback instruction must appear IN THE PROMPT  not just in the answer.
    Tests that we control the instruction, not just observe the LLM's behavior.
    """
    with patch("backend.app.rag.pipeline.OllamaLLM") as mock_llm_class, \
         patch("backend.app.rag.pipeline.Reranker") as mock_reranker_class:

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "I don't have enough information in the available documents "
            "to answer this question."
        )
        mock_llm_class.return_value = mock_llm
        mock_reranker_class.return_value = MagicMock()

        pipeline = RAGPipeline(seeded_vector_store, embedding_pipeline)
        pipeline.retriever.retrieve = MagicMock(return_value=[])
        response = pipeline.run(RAGRequest(question="What is the CEO salary?"))

    actual_prompt = mock_llm.invoke.call_args[0][0]
    assert "I don't have enough information" in actual_prompt
    assert response.retrieval_count == 0
    assert response.sources == []
    assert "don't have enough information" in response.answer.lower()
    mock_llm.invoke.assert_called_once()