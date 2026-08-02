from fastapi import FastAPI

from backend.app.api.documents import router as documents_router
from backend.app.api.rag import router as rag_router

app = FastAPI(title="Ragnarok RAG Platform", version="0.1.0")

app.include_router(documents_router)
app.include_router(rag_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
