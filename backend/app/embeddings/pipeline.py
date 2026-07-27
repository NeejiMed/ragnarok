from sentence_transformers import SentenceTransformer # SentenceTransformer is a class from the sentence_transformers library that allows you to load pre-trained models for generating sentence embeddings. It provides an easy-to-use interface for encoding sentences into fixed-size vectors, which can be used for various natural language processing tasks such as semantic similarity, clustering, and classification.
from typing import cast

from backend.app.chunking.schemas import DocumentChunk 
from backend.app.embeddings.schemas import EmbeddedChunk 

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5" 

class EmbeddingPipeline:
    """
    Converts DocumentChunks into EmbeddedChunks by running the embedding model.
    Designed as a class (not module-level functions) so the model is loaded only once
    at instantiation and reused across all embed() calls. That avoids reloading the model on every request.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)  # Load the pre-trained model specified by model_name. This model will be used to generate embeddings for the document chunks.

    def embed(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        """
        Embeds a list of DocumentChunks in a single batched model call.
        Returns EmbeddedChunks with the embedding vector and model name added.
        """
        if not chunks:
            return []
        
        texts = [ chunk.content for chunk in chunks ]

        # Single batched encode call, dramatically faster than one call per chunk
        vectors = self.model.encode(
            texts,
            batch_size=32,  # Process 32 texts at a time for efficiency
            show_progress_bar=False,  # Disable progress bar for cleaner output
            normalize_embeddings=True  # L2 normalize -> cosine similarity = dot product (faster Qdrant search)
        )

        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                strategy=chunk.strategy,
                metadata=chunk.metadata,
                embedding=vector.tolist(),  # Convert the numpy array to a list for JSON serialization
                embedding_model=self.model_name
            )
            for chunk, vector in zip(chunks, vectors)
        ]
    
    @property # this decorator allows the method to be accessed like an attribute, without needing to call it as a function.
    def embedding_dimension(self) -> int:
        """
        Returns the vector dimension for the current model. This is needed by Qdrant"
        """
        return cast(int, self.model.get_embedding_dimension()) # Retrieve the dimensionality of the embeddings produced by the model. This is important for downstream tasks like storing embeddings in a vector database (e.g., Qdrant) that requires knowledge of the embedding size.
    