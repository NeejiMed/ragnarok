from typing import TypedDict

class RAGAgentState(TypedDict):
    """
    State object passed between all nodes in the RAG agent graph.
    Every node receives this, modifies relevant fields, and returns it.
    """
    question: str
    query: str
    retrieved_chunks: list[dict]
    context: str
    answer: str
    validation_result: str  # "pass" or "fail" or "pending"
    retry_count: int
    max_retries: int