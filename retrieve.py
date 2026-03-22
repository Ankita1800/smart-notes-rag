"""
retrieve.py
-----------
RAG retrieval pipeline: converts a question into an embedding,
fetches the top-3 relevant chunks from Endee, and formats a concise answer.
"""

import re
import logging
import numpy as np
from utils import embed_query, embed_texts
from endee_store import search_chunks
from llm_optional import enhance_answer

logger = logging.getLogger(__name__)

# How many top sentences to pull from each chunk
TOP_SENTENCES_PER_CHUNK = 2


def answer_question(question: str, top_k: int = 3, use_llm: bool = False) -> dict:
    """
    Full RAG pipeline:
      1. Embed the user question.
      2. Search Endee for top-k relevant chunks.
      3. From each chunk, extract the 2 most relevant sentences.
      4. Return a clean, concise, formatted answer.
    """
    if not question.strip():
        return {
            "question": question,
            "answer": "Please provide a non-empty question.",
            "sources": [],
            "llm_used": False,
        }

    logger.info("Processing question: %s", question)

    query_vector = embed_query(question)
    hits = search_chunks(query_vector, top_k=top_k)

    if not hits:
        return {
            "question": question,
            "answer": "No relevant content found. Please upload notes first.",
            "sources": [],
            "llm_used": False,
        }

    if use_llm:
        answer, llm_used = enhance_answer(question, hits)
    else:
        answer = _build_concise_answer(question, query_vector, hits)
        llm_used = False

    return {
        "question": question,
        "answer": answer,
        "sources": hits,
        "llm_used": llm_used,
    }


# ── Core answer builder ───────────────────────────────────────────────────────

def _build_concise_answer(question: str, query_vector: list[float], hits: list[dict]) -> str:
    """
    Instead of dumping full 300-word chunks, this function:
      1. Splits each chunk into individual sentences.
      2. Embeds all sentences and ranks them by cosine similarity to the question.
      3. Picks the top 2 most relevant sentences per chunk.
      4. Deduplicates and formats them into a clean, readable answer.
    """
    q_vec = np.array(query_vector, dtype=np.float32)

    answer_points: list[dict] = []   # {sentence, score, source_rank}

    for rank, hit in enumerate(hits):
        sentences = _split_sentences(hit["text"])
        if not sentences:
            continue

        # Embed all sentences in this chunk at once
        sent_embeddings = embed_texts(sentences)

        # Score each sentence against the query
        scored = []
        for sent, emb in zip(sentences, sent_embeddings):
            s_vec = np.array(emb, dtype=np.float32)
            score = float(np.dot(q_vec, s_vec))   # both normalised → cosine sim
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top sentences that are long enough to be informative
        taken = 0
        for score, sent in scored:
            sent = sent.strip()
            if len(sent.split()) < 6:   # skip trivially short sentences
                continue
            answer_points.append({
                "sentence": sent,
                "score": score,
                "source_rank": rank,
            })
            taken += 1
            if taken >= TOP_SENTENCES_PER_CHUNK:
                break

    if not answer_points:
        # Fallback: first sentence of best chunk
        first_sent = _split_sentences(hits[0]["text"])
        return first_sent[0] if first_sent else hits[0]["text"][:300]

    # Globally sort all collected sentences by relevance
    answer_points.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate near-identical sentences
    seen: list[str] = []
    for ap in answer_points:
        if not _is_duplicate(ap["sentence"], seen):
            seen.append(ap["sentence"])

    # Format: bullet points, max 5 sentences total
    final_sentences = seen[:5]

    lines = []
    for sent in final_sentences:
        # Ensure sentence ends with punctuation
        if sent and sent[-1] not in ".!?":
            sent += "."
        lines.append(f"• {sent}")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into individual sentences."""
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    # Also split on newlines that act as sentence breaks
    sentences = []
    for part in raw:
        sub = [s.strip() for s in part.split('\n') if s.strip()]
        sentences.extend(sub)
    return [s for s in sentences if s]


def _is_duplicate(candidate: str, seen: list[str], threshold: float = 0.85) -> bool:
    """
    Simple word-overlap deduplication.
    Returns True if *candidate* is too similar to any sentence already in *seen*.
    """
    cand_words = set(candidate.lower().split())
    for s in seen:
        s_words = set(s.lower().split())
        if not cand_words or not s_words:
            continue
        overlap = len(cand_words & s_words) / max(len(cand_words), len(s_words))
        if overlap >= threshold:
            return True
    return False
