import sys
from unittest.mock import MagicMock

sys.modules["qdrant_client.connection"] = MagicMock()
sys.modules["qdrant_client.async_qdrant_remote"] = MagicMock()
sys.modules["qdrant_client.async_qdrant_client"] = MagicMock()

# qdrant_client uses isinstance(point, grpc.PointStruct) at runtime
# MagicMock instances can't be used as isinstance() second arg
# so we provide real classes for the types that get isinstance-checked
_qdrant_grpc = MagicMock()


class _FakePointStruct:
    pass


class _FakeBatch:
    pass


_qdrant_grpc.PointStruct = _FakePointStruct
_qdrant_grpc.Batch = _FakeBatch
sys.modules["qdrant_client.grpc"] = _qdrant_grpc

import pytest
from qdrant_client import QdrantClient

from backend.app.embeddings.pipeline import EmbeddingPipeline
from backend.app.vectorstore.store import VectorStore


@pytest.fixture(scope="session")
def in_memory_qdrant_client():
    """
    In-memory Qdrant — no Docker, no network, no grpc DLL.
    Uses QdrantLocal internally which bypasses all transport modules.
    """
    return QdrantClient(":memory:")


@pytest.fixture(scope="session")
def embedding_pipeline():
    """Embedding model loaded once per session, shared across all test files."""
    return EmbeddingPipeline()