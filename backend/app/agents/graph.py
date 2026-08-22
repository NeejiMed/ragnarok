from typing import cast

from langgraph.graph import END, START, StateGraph

from backend.app.agents.nodes import (
    make_answer_node,
    make_retrieve_node,
    make_retry_node,
    should_retry,
    validate_node,
)
from backend.app.agents.state import RAGAgentState
from backend.app.embeddings.pipeline import EmbeddingPipeline
from backend.app.vectorstore.store import VectorStore


def build_rag_graph(
    vector_store: VectorStore,
    embedding_pipeline: EmbeddingPipeline,
    llm_model: str = "llama3.2",
    top_k: int = 5,
    max_retries: int = 2,
):
    """
    Builds and compiles the RAG agent graph.
    Returns a compiled LangGraph app ready to invoke.
    """
    retrieve_node = make_retrieve_node(vector_store, embedding_pipeline, top_k=top_k)
    answer_node = make_answer_node(llm_model=llm_model)
    retry_node = make_retry_node()

    graph = StateGraph(RAGAgentState)

    # add nodes to the graph
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)

    # define edges between nodes
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "validate")

    # conditional edges based on validation result
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {
            "retry": "retry",
            "done": END,
        },
    )

    # After retry, go back to retrieve
    graph.add_edge("retry", "retrieve")

    return graph.compile()


def run_rag_agent(
    question: str,
    vector_store: VectorStore,
    embedding_pipeline: EmbeddingPipeline,
    llm_model: str = "llama3.2",
    top_k: int = 5,
    max_retries: int = 2,
) -> RAGAgentState:
    """
    Convenience function to build the graph and run it for one question.
    returns the final state object after the graph completes.
    """
    app = build_rag_graph(vector_store, embedding_pipeline, llm_model, top_k, max_retries)

    initial_state: RAGAgentState = {
        "question": question,
        "query": question,
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "validation_result": "pending",
        "retry_count": 0,
        "max_retries": max_retries,
    }

    return cast(RAGAgentState, app.invoke(initial_state))
