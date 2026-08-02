from pydantic import BaseModel
from typing import Any

class RAGRequest(BaseModel):
    """ Incoming RAG query from the user. """
    question: str
    top_k: int = 5
    filters: dict[str,Any] | None = None
    use_reranking: bool = False

class RetrievedSource(BaseModel):
    """ A single retrieved chunk with its citation metadata. """
    content: str
    score: float
    source: str | None = None
    page: int | None = None
    document_id: str | None = None

class RAGResponse(BaseModel):
    """ Complete RAG pipeline response, including the answer and the retrieved sources. """
    question: str
    answer: str
    sources: list[RetrievedSource]
    retrieval_count: int
    model_used: str