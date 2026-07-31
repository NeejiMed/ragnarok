import numpy as np
import pytest

from backend.app.chunking.schemas import DocumentChunk
from backend.app.embeddings.pipeline import EmbeddingPipeline


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


@pytest.fixture(scope="module")
def embedding_pipeline():
    # scope="module" ensures the model loads once for all tests in this file
    return EmbeddingPipeline()


@pytest.fixture
def document_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="The cat sat on the mat.",
            chunk_index=0,
            strategy="test",
            metadata={"source": "test"},
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="A cat is sitting on a mat.",
            chunk_index=1,
            strategy="test",
            metadata={"source": "test"},
        ),
    ]


@pytest.fixture
def embedded_chunks(embedding_pipeline, document_chunks):
    return embedding_pipeline.embed(document_chunks)


@pytest.mark.slow
def test_embed_returns_correct_count(embedded_chunks, document_chunks):
    assert len(embedded_chunks) == len(document_chunks)


@pytest.mark.slow
def test_embedded_chunk_fields_match_input(embedded_chunks, document_chunks, embedding_pipeline):
    for embedded, original in zip(embedded_chunks, document_chunks, strict=False):
        assert embedded.chunk_id == original.chunk_id
        assert embedded.document_id == original.document_id
        assert embedded.content == original.content
        assert embedded.chunk_index == original.chunk_index
        assert embedded.strategy == original.strategy
        assert embedded.metadata == original.metadata
        assert embedded.embedding_model == embedding_pipeline.model_name


@pytest.mark.slow
def test_embedding_dimensions(embedded_chunks, embedding_pipeline):
    expected_dimension = embedding_pipeline.embedding_dimension  # singular
    for chunk in embedded_chunks:
        assert isinstance(chunk.embedding, list)
        assert all(isinstance(x, float) for x in chunk.embedding)
        assert len(chunk.embedding) == expected_dimension


@pytest.mark.slow
def test_embed_empty_list(embedding_pipeline):
    assert embedding_pipeline.embed([]) == []


@pytest.mark.slow
def test_semantic_similarity_sanity(embedding_pipeline):
    """Similar sentences should have higher cosine similarity than dissimilar ones."""

    def make_chunk(chunk_id, doc_id, text, index):
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            content=text,
            chunk_index=index,
            strategy="test",
            metadata={},
        )

    similar = embedding_pipeline.embed(
        [
            make_chunk("s1", "doc1", "The cat sat on the mat.", 0),
            make_chunk("s2", "doc1", "A cat is sitting on a mat.", 1),
        ]
    )

    dissimilar = embedding_pipeline.embed(
        [
            make_chunk("d1", "doc2", "The cat sat on the mat.", 0),
            make_chunk("d2", "doc2", "The sun is shining brightly.", 1),
        ]
    )

    sim_score = cosine_similarity(similar[0].embedding, similar[1].embedding)
    dissim_score = cosine_similarity(dissimilar[0].embedding, dissimilar[1].embedding)

    assert (
        sim_score > dissim_score
    ), f"Expected similar pair ({sim_score:.3f}) > dissimilar pair ({dissim_score:.3f})"
