"""
LLM integration for grounded answer generation.
Uses OpenAI-compatible API (OpenAI, Azure, Ollama, etc.).
"""

import logging
from typing import TYPE_CHECKING

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

if TYPE_CHECKING:
    from search.store import SearchResult

logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    """Create OpenAI client with optional custom base URL."""
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def generate_answer(query: str, results: list["SearchResult"]) -> str:
    """
    Generate a grounded answer using retrieved chunks.
    
    Args:
        query: User question
        results: List of SearchResult from vector store
    
    Returns:
        Answer string grounded in the documents
    """
    if not results:
        return "No relevant documents found to answer this question."

    # Build context from top results
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(f"[Document {i}: {result.file_name}]\n{result.text}")

    context = "\n\n".join(context_parts)

    # If mock mode is enabled, return context directly
    if settings.use_mock_llm:
        logger.info("Using mock LLM mode (no OpenAI API call)")
        return f"Based on the provided documents:\n\n{context[:500]}..."

    # Grounded prompt — forces LLM to answer only from context
    system_prompt = """You are a helpful assistant that answers questions based strictly on provided documents.

IMPORTANT RULES:
1. Answer ONLY using information from the provided documents.
2. If the answer is not in the documents, say "I cannot find this information in the provided documents."
3. Do NOT make up, infer, or use external knowledge.
4. Be concise and direct.
5. If multiple documents are relevant, synthesize them naturally."""

    user_message = f"""Based on the following documents, answer this question:

QUESTION: {query}

DOCUMENTS:
{context}

ANSWER:"""

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        answer = response.choices[0].message.content or ""
        logger.info("Generated answer for query: %s", query[:50])
        return answer.strip()

    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise
