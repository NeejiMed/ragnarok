from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant for answering questions about company documents.
Answer the question based ONLY on the context provided below.
If the answer cannot be found in the context, respond with:
"I don't have enough information in the available documents to answer this question.
Do not make up information or use knowledge outsied the provided context.

context:
{context}

Question: {question}

Answer:"""
)

def format_context(results: list[dict]) -> str:
    """
    Formats retrieved chunks into a  readable context string for the LLM.
    Includes source citations so the LLM can reference them in its answer.
    """
    context_parts = []
    for i, result in enumerate(results, start=1):
        source = result.get("source", "Unknown")
        page = result.get("page")
        content = result.get("content", "")

        citation = f"[Source {i}: {source}"
        if page is not None:
            citation += f", Page {page}"
        citation += "]"

        context_parts.append(f"{citation}\n{content}")

    return "\n\n".join(context_parts)