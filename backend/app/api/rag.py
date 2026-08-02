from fastapi import APIRouter, HTTPException

from backend.app.rag.schemas import RAGRequest, RAGResponse
from backend.app.rag.pipeline import RAGPipeline
from backend.app.core.config import settings
from backend.app.embeddings.pipeline import EmbeddingPipeline
from backend.app.vectorstore.store import VectorStore

router = APIRouter(prefix="/rag", tags=["rag"]) # this router will handle all RAG-related endpoints

# Module-level singleton: initialized once when the models loads
# In a production system these would be dependency-injected via FastAPI's Depends()
_vector_store: VectorStore | None = None
_embedding_pipeline: EmbeddingPipeline | None = None
_rag_pipeline: RAGPipeline | None = None

def get_rag_pipeline() -> RAGPipeline:
    global _vector_store, _embedding_pipeline, _rag_pipeline
    if _rag_pipeline is None:
        _vector_store = VectorStore()
        _vector_store.ensure_collection(
            embedding_dimension=384
        )
        _embedding_pipeline = EmbeddingPipeline()
        _rag_pipeline = RAGPipeline(_vector_store, _embedding_pipeline)
    return _rag_pipeline


@router.post("/query", response_model=RAGResponse)
async def query(request: RAGRequest) -> RAGResponse:
    try:
        pipeline = get_rag_pipeline()
        return pipeline.run(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e