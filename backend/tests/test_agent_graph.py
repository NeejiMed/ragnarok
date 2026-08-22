from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from backend.app.agents.graph import run_rag_agent

#  Pure unit tests: should_retry
# These test a pure Python function  no fixtures, no mocks, instant.
from backend.app.agents.nodes import should_retry, validate_node
from backend.app.agents.state import RAGAgentState
from backend.app.chunking.schemas import DocumentChunk
from backend.app.vectorstore.store import VectorStore

#  Helpers


def make_state(**overrides) -> RAGAgentState:
    """
    Builds a minimal valid RAGAgentState with sensible defaults.
    Pass keyword args to override specific fields.
    Centralizing state construction avoids repetition and makes
    tests resilient to future state schema changes.
    """
    base: RAGAgentState = {
        "question": "What is the refund policy?",
        "query": "What is the refund policy?",
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "validation_result": "pending",
        "retry_count": 0,
        "max_retries": 2,
    }
    # typing: build dict then cast to RAGAgentState to satisfy static type checkers
    return cast(RAGAgentState, {**base, **overrides})


@pytest.fixture
def seeded_vector_store(in_memory_qdrant_client, embedding_pipeline):
    """
    Vector store pre-seeded with one domain-realistic chunk.
    Used for full graph integration tests.
    """

    store = VectorStore(
        collection_name=f"test_agent_{uuid4().hex[:8]}",
        client=in_memory_qdrant_client,
    )
    store.ensure_collection(embedding_dimension=384)

    chunks = [
        DocumentChunk(
            chunk_id=str(uuid4()),
            document_id="refund-policy",
            content="The company refund policy allows returns within 30 days.",
            chunk_index=0,
            strategy="test",
            metadata={"source": "policy-handbook", "page": 3},
        )
    ]
    embedded = embedding_pipeline.embed(chunks)
    store.upsert(embedded)
    return store


def test_should_retry_returns_done_when_pass():
    """Passing validation always ends the graph regardless of retry count."""
    state = make_state(validation_result="pass", retry_count=0)
    assert should_retry(state) == "done"


def test_should_retry_returns_retry_when_fail_and_retries_remain():
    """Failed validation with retries remaining should loop back."""
    state = make_state(validation_result="fail", retry_count=0, max_retries=2)
    assert should_retry(state) == "retry"


def test_should_retry_returns_done_when_max_retries_reached():
    """
    Circuit breaker: even on failure, stop when retry_count >= max_retries.
    Prevents infinite loops  critical for production agentic systems.
    """
    state = make_state(validation_result="fail", retry_count=2, max_retries=2)
    assert should_retry(state) == "done"


def test_should_retry_returns_done_when_retries_exceeded():
    """retry_count > max_retries (e.g. due to a bug) must also terminate."""
    state = make_state(validation_result="fail", retry_count=5, max_retries=2)
    assert should_retry(state) == "done"


#  Pure unit tests: validate_node


def test_validate_node_passes_confident_answer():
    """A specific, factual answer should pass validation."""
    state = make_state(answer="The refund policy allows returns within 30 days.")
    result = validate_node(state)
    assert result["validation_result"] == "pass"


def test_validate_node_fails_on_i_dont_know():
    """'I don't know' signal must trigger validation failure."""
    state = make_state(answer="I don't know the answer to this question.")
    result = validate_node(state)
    assert result["validation_result"] == "fail"


def test_validate_node_fails_on_no_information():
    """'I don't have enough information' is a known low-confidence signal."""
    state = make_state(answer="I don't have enough information in the available documents.")
    result = validate_node(state)
    assert result["validation_result"] == "fail"


def test_validate_node_fails_on_cannot_find():
    """'cannot find' is a known low-confidence signal."""
    state = make_state(answer="I cannot find any relevant information about this.")
    result = validate_node(state)
    assert result["validation_result"] == "fail"


def test_validate_node_preserves_all_state_fields():
    """
    validate_node must return the full state, not just validation_result.
    A node that drops fields silently corrupts the graph's state.
    """
    state = make_state(
        answer="The refund policy is 30 days.",
        question="What is the refund policy?",
        retry_count=1,
    )
    result = validate_node(state)

    assert result["question"] == "What is the refund policy?"
    assert result["retry_count"] == 1
    assert result["answer"] == "The refund policy is 30 days."
    assert result["validation_result"] == "pass"


#  Integration + slow: full graph execution


@pytest.mark.slow
def test_full_graph_confident_answer_no_retry(seeded_vector_store, embedding_pipeline):
    """
    Full graph run: confident LLM answer should pass validation
    and complete in exactly one LLM call (no retry triggered).
    """
    with patch("backend.app.agents.nodes.OllamaLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The refund policy allows returns within 30 days."
        mock_llm_class.return_value = mock_llm

        result = run_rag_agent(
            question="What is the refund policy?",
            vector_store=seeded_vector_store,
            embedding_pipeline=embedding_pipeline,
        )

    assert result["validation_result"] == "pass"
    assert result["answer"] == "The refund policy allows returns within 30 days."
    assert result["retry_count"] == 0
    assert mock_llm.invoke.call_count == 1


@pytest.mark.slow
def test_full_graph_low_confidence_triggers_retry(seeded_vector_store, embedding_pipeline):
    """
    When LLM returns a low-confidence answer on first call, the graph
    should retry. On the second call it returns a confident answer.
    Verifies the retry loop actually executes.
    """
    call_count = {"n": 0}

    def side_effect(prompt):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "I don't have enough information to answer this question."
        return "The refund policy allows returns within 30 days."

    with patch("backend.app.agents.nodes.OllamaLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = side_effect
        mock_llm_class.return_value = mock_llm

        result = run_rag_agent(
            question="What is the refund policy?",
            vector_store=seeded_vector_store,
            embedding_pipeline=embedding_pipeline,
            max_retries=2,
        )

    assert result["retry_count"] == 1
    assert result["validation_result"] == "pass"
    assert mock_llm.invoke.call_count == 2


@pytest.mark.slow
def test_full_graph_circuit_breaker_stops_infinite_retry(seeded_vector_store, embedding_pipeline):
    """
    When LLM always returns low-confidence answers, the circuit breaker
    (max_retries) must terminate the graph instead of looping forever.
    retry_count should equal max_retries at termination.
    """
    with patch("backend.app.agents.nodes.OllamaLLM") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "I don't know anything about this topic."
        mock_llm_class.return_value = mock_llm

        result = run_rag_agent(
            question="What is the quantum entanglement policy?",
            vector_store=seeded_vector_store,
            embedding_pipeline=embedding_pipeline,
            max_retries=2,
        )

    assert result["retry_count"] == 2
    assert result["validation_result"] == "fail"
    assert mock_llm.invoke.call_count == 3  # initial + 2 retries
