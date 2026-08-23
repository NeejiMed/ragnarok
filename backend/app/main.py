from fastapi import FastAPI
from prometheus_client import make_asgi_app

from backend.app.api.documents import router as documents_router
from backend.app.api.evaluation import router as evaluation_router
from backend.app.api.rag import router as rag_router
from backend.app.core.middleware import prometheus_middleware

app = FastAPI(title="Ragnarok RAG Platform", version="0.1.0")

# Prometheus middleware — must be added before routers
app.middleware("http")(prometheus_middleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(evaluation_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
