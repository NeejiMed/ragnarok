from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from backend.app.retrieval.retriever import Retriever

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question", "n"],
    template="""You are an AI assistant helping improve document retrieval.
Generate {n} different versions of the following question to retrieve
relevant documents from a vector database. Each version should approach
the question from a different angle or use different vocabulary.
Return only the questions, one per line, no numbering or extra text.input_types=

Original question: {question}

Alternative questions:""",
)


class MultiQueryRetriever:
    """
    Layer 2: Generate multiple query variations using an LLM,
    and retrieves chunks for each, and deduplicates results.
    Requires an LLM that can run Ollama to avoid sending sensitive queries to a cloud API.
    """

    def __init__(self, retriever: Retriever, model_name: str = "llama3", n_queries: int = 3):
        self.retriever = retriever
        self.llm = OllamaLLM(model=model_name)
        self.n_queries = n_queries

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        """
        Generates multiple query variations and retrieves relevant chunks for each.
        Deduplicates results based on chunk_id.
        Returns a list of payload dicts with similarity score.
        """
        # Generate alternative queries using the LLM
        prompt = MULTI_QUERY_PROMPT.format(question=query, n=self.n_queries)
        response = self.llm.invoke(prompt)
        alternative_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]

        # Always include the original query as well
        all_queries = [query] + alternative_queries[: self.n_queries]

        # Retrieve for each query, deduplicate by chunk_id
        seen_ids = set()
        merged_results = []

        for q in all_queries:
            results = self.retriever.retrieve(q, top_k=top_k, filters=filters)
            for result in results:
                chunk_id = result.get("chunk_id")
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    merged_results.append(result)

        # Sort merged results by score in descending order
        merged_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged_results
