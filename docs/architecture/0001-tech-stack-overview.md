### Frontend: Streamlit
**Alternatives considered:** Gradio, plain HTML/JS, React/Next.js
**Why Streamlit:** Fastest path to an interactive AI prototype with pure Python - no separate
frontend build step, tight integration with my backend skillset (Python-first).
**Trade-off accepted:** Not suitable for production-scale multi-user UI (no fine-grained
state control, weaker custom styling). We explicitly scope this as an MVP/internal-tool
frontend, to be replaced by a dedicated frontend framework if this becomes a real product.
**Revisit when:** we need multi-tenant auth, custom UX, or mobile support.

# Backend: FastAPI
**Alternatives considered:** Flask, Django REST Framework, Sanic
**Why FastAPI:** Used for  high-performance python web framework, It helps in building restful APIs, microservices and data driven web application, quick to build with a data-focused web app
**Trade-off accepted:** Relies deeply on Pydantic for data validation, and offers fewer third-party plugins compared to Django or Flask

**Alternatives considered for Uvicorn:** Hypercorn, NGINX Unit
**Why Uvicorn:** Fimiliar with, and very useful to run modern, high-concurrency python web applications and APIs as it's the default native server for FastAPI
**Trade-off accepted:**  Uvicorn lacks native HTTP/3 support out-of-the-box


**Alternatives considered for Pydantic:** Flask, Django REST Framework, Sanic
**Why Pydantic:** It gurantees runtime data validation and parsing using native python type hints
**Trade-off accepted:** Relies deeply on Pydantic for data validation, and offers fewer third-party plugins compared to Django or Flask

# Vector Database: Qdrant
**Alternatives considered:** Pinecone(managed, paid), Weaviate(Similar to Qdrant), FAISS(library, not a DB & no built-in persistence/filtering/server), Chroma(simpler but less production grade)
**Why Qdrant:** free, self-hostable, native hybrid search + metadata filtering, production-grade HNSW indexing, good Python client, Docker friendly
**Trade-off accepted:** extra moving part to deploy/monitor vs an embedded library like Chroma/FAISS

# LLM Framework (LangChain)
**Alternatives considered:** raw API calls + custom orchestration, LlamaIndex, Haystack
**Why LangChain:** Native graph-based control flow for multi-agent validation pipeline
**Trade-off accepted:** can add undessary abstraction overhead for simple use cases

# LLM Runtime: Ollama
**Alternatives considered:** OpenAI API, Anthropic API, Google Gemini API, Hugging Face Inference Endpoints
**Why Ollama:** Runs models locally, is free to use, protects sensitive data by keeping it on local machine and makes development possible without API costs
**Trade-off accepted:** Local models generally provide lower quality responses and slower inference than the latest hosted commercial models and they require local hardware resources.
**Revist when:** Higher-quality responses, larger context windows, better latency ir enterprise-grade reliability become more important than avoiding API costs.

# Experiment Tracking: MLflow
**Alternatives considered:** Weights & Biases (W&B), Neptune.ai, TensorBoard
**Why MLflow:** Open source, widely adopted in MLOps, easy to self-host, and provides experiment tracking without depending on a commercial cloud service.
**Trade-off accepted:** Requires hosting and maintaining your own tracking server, whereas managed platforms reduce operational work.
**Revisit when:** The team grows, collaboration becomes more important, or advanced experiment visualization and reporting justify using a managed platform.

# Monitoring: Prometheus + Grafana

**Alternatives considered:** Datadog, New Relic, Elastic Stack (ELK)
**Why Prometheus + Grafana:** Industry-standard open-source monitoring stack. Prometheus collects metrics, while Grafana provides flexible dashboards for visualizing application and infrastructure health.
**Trade-off accepted:** More infrastructure to deploy and maintain compared to fully managed monitoring services.
**Revisit when:** The system moves to a cloud environment where managed monitoring solutions reduce operational overhead or provide features the project requires.

# Infrastructure: Docker Compose

**Alternatives considered:** Running everything directly on the host machine, Kubernetes, Podman
**Why Docker Compose:** Simple way to run all project services together during development. Easy to understand, quick to set up, and sufficient for a portfolio project and MVP.
**Trade-off accepted:** Limited scalability, high availability, and orchestration capabilities compared to Kubernetes.
**Revisit when:** The application needs multiple replicas, automatic scaling, rolling deployments, or production-grade orchestration.
