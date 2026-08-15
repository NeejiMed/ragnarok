from langchain_ollama import OllamaLLM

from backend.app.agents.state import RAGAgentState
from backend.app.embeddings.pipeline import EmbeddingPipeline
from backend.app.rag.prompt import RAG_PROMPT, format_context
from backend.app.vectorstore.store import VectorStore

def make_retrieve_node(
        vector_store: VectorStore,
        embedding_pipeline: EmbeddingPipeline,
        top_k: int = 5
):
    """
    Factory function that returns a retieve_node function bound to the
    provided vector_store and embedding_pipeline instances. 
    Using a factory avoids  global state while keeping node signatures
    compatible with LangGraph's (state) -> State contract.
    """
    from backend.app.chunking.schemas import DocumentChunk
    from uuid import uuid4

    def retrieve_node(state: RAGAgentState) -> RAGAgentState:
        """Embeds the current query and retrieves top-k chunks from Qdrant."""
        query_chunk = DocumentChunk(
            chunk_id=str(uuid4()),
            document_id="query",
            content=state["query"],
            chunk_index=0,
            strategy="query",
            metadata={}
        )
        embedded = embedding_pipeline.embed([query_chunk])
        query_vector = embedded[0].embedding

        results = vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )
        return {
            **state,
            "retrieved_chunks": results,
            "context": format_context(results)
        }

    return retrieve_node

def make_answer_node(llm_model: str = "llama3.2"):
    """Factory returns an answer_node bound to the specified LLM model."""

    llm = OllamaLLM(model=llm_model)

    def answer_node(state: RAGAgentState) -> RAGAgentState:
        """Formats context + question into a prompt and calls the LLM"""
        prompt = RAG_PROMPT.format(
            context=state["context"],
            question=state["question"]
        )
        answer = llm.invoke(prompt)
        return {
            **state,
            "answer": answer
        }

    return answer_node

def validate_node(state: RAGAgentState) -> RAGAgentState:
    """
    Validates answer quality by checking for known low-confidence signals.
    Sets validation_result to "pass", "fail".
    "fail" triggers a retry if retry_count < max_retries.
    """
    answer = state["answer"].lower()
    no_info_signals = [
        "i don't know",
        "i don't have enough information",
        "cannot find",
        "not available in the",
        "no information",
        "not enough information"]   

    failed = any(signal in answer for signal in no_info_signals)
    return {
        **state,
        "validation_result": "fail" if failed else "pass"
    }

def should_retry(state: RAGAgentState) -> str:
    """
    conditional edge function that returns the name of the next node.
    LangGraph calls this after validate_node to decide routing.
    Returns "retry" to loop back to retrieve_node, or "done" to exit the graph.
    """
    if state["validation_result"] == "fail" and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "done"

def make_retry_node():
    """
    On retry: increment retry_count and expand the query slightly.
    """
    def retry_node(state: RAGAgentState) -> RAGAgentState:
        return {
            **state,
            "retry_count": state["retry_count"] + 1,
            "query": state['query']
        }
    return retry_node