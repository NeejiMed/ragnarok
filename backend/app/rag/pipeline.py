from langchain_ollama import OllamaLLM

from backend.app.core.config import settings
from backend.app.embeddings.pipeline import EmbeddingPipeline
from backend.app.rag.prompt import RAG_PROMPT, format_context
from backend.app.rag.schemas import RAGRequest, RAGResponse, RetrievedSource
from backend.app.retrieval.reranker import Reranker
from backend.app.retrieval.retriever import Retriever
from backend.app.vectorstore.store import VectorStore

class RAGPipeline:
    """
    Orchestrates the full RAG flow:
    question -> retrieval -> context -> prompt -> LLM -> answer
    Instantiated once at startup and reused across requests.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        embedding_pipeline: EmbeddingPipeline,
        llm_model: str = "llama3.2"
    ):
        self.retriever = Retriever(vectorstore, embedding_pipeline)
        self.reranker = Reranker()
        self.llm = OllamaLLM(model=llm_model)
        self.llm_model = llm_model

    def run(self, request: RAGRequest) -> RAGResponse:
        """
        Executes the full RAG pipeline synchronously for a given RAGRequest.
        """
        # Step 1: Retrieve relevant chunks 
        results = self.retriever.retrieve(
            query=request.question,
            top_k=request.top_k * 2 if request.use_reranking else request.top_k,
            filters=request.filters
        )

        # Step 2: Optionally rerank the retrieved chunks
        if request.use_reranking and results: # Only rerank if there are results to rerank
            results = self.reranker.rerank(
                query=request.question,
                results=results,
                top_k=request.top_k
            )

        # Step 3: Format the retrieved chunks into a context string
        context = format_context(results)

        # Step 4: build and invoke the prompt with the context and question
        prompt = RAG_PROMPT.format(context=context, question=request.question)
        answer = self.llm.invoke(prompt)

        # Step 5: Build the response object with the source citations
        sources = [
            RetrievedSource(
                content=r.get("content", ""),
                score=r.get("rerank_score", r.get("score", 0.0)),
                source=r.get("source"),
                page=r.get("page"),
                document_id=r.get("document_id")
            )
            for r in results
        ]

        return RAGResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            retrieval_count=len(results),
            model_used=self.llm_model
        )