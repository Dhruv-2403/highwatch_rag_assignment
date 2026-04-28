from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from search.store import SearchResult

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


SYSTEM_PROMPT = """You are an AI Expert Assistant that answers questions
strictly based on the provided document context.

Rules:
1. Answer ONLY from the provided context. Do not hallucinate.
2. If the context doesn't contain enough information, say so clearly.
3. Be concise and precise.
4. Always cite which document(s) your answer comes from.
5. Use plain, professional English.
"""

CONTEXT_TEMPLATE = """<context>
{context_blocks}
</context>

Question: {question}

Answer based only on the context above. If the answer is not in the context, reply:
"I couldn't find relevant information in the available documents."
"""


def _build_context(results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[{i}] Source: {r.file_name}\n"
            f"Relevance score: {r.score:.3f}\n"
            f"---\n{r.text}"
        )
    return "\n\n".join(blocks)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_answer(query: str, results: list[SearchResult]) -> str:
    
    if not results:
        return "No relevant documents found. Please sync your Drive first or rephrase your question."

    context = _build_context(results)
    user_message = CONTEXT_TEMPLATE.format(context_blocks=context, question=query)

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""
