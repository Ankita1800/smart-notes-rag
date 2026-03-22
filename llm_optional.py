"""
llm_optional.py
---------------
OPTIONAL: Enhance the RAG answer using an LLM.

This module is entirely optional.
If no LLM is configured, the system falls back gracefully to returning
the raw retrieved chunks — the core project works without any API key.

To enable:
    1. Install: pip install openai   (or any LLM provider you prefer)
    2. Set env var: OPENAI_API_KEY=sk-...
    3. Pass use_llm=True to answer_question() in retrieve.py

The function signature is fixed:
    enhance_answer(question: str, hits: list[dict]) -> tuple[str, bool]
    Returns (answer_text, llm_was_used)
"""

import os
import logging

logger = logging.getLogger(__name__)


def enhance_answer(question: str, hits: list[dict]) -> tuple[str, bool]:
    """
    Attempt to generate an LLM-enhanced answer.

    Falls back to raw chunk formatting if:
    - The openai package is not installed
    - OPENAI_API_KEY is not set
    - The API call fails for any reason

    Returns (answer_string, llm_was_used_bool)
    """
    try:
        import openai  # noqa: F401 – optional dependency
    except ImportError:
        logger.info("openai not installed – falling back to raw retrieval.")
        return _raw_fallback(question, hits), False

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.info("OPENAI_API_KEY not set – falling back to raw retrieval.")
        return _raw_fallback(question, hits), False

    # Build context from retrieved chunks
    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(f"[Chunk {i}]\n{hit['text'].strip()}")
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful study assistant. Answer the question below using ONLY "
        "the provided context from the user's notes. Be concise and accurate. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("LLM answer generated successfully.")
        return answer, True
    except Exception as exc:
        logger.warning("LLM call failed (%s) – falling back to raw retrieval.", exc)
        return _raw_fallback(question, hits), False


def _raw_fallback(question: str, hits: list[dict]) -> str:
    """Format a clean answer from raw chunks (no LLM)."""
    lines = [
        f'Most relevant sections for: "{question}"\n',
    ]
    for i, hit in enumerate(hits, start=1):
        score_pct = int(hit["similarity"] * 100)
        lines.append(f"── Section {i}  (relevance: {score_pct}%) ──")
        lines.append(hit["text"].strip())
        lines.append("")
    return "\n".join(lines).strip()
