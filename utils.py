"""
utils.py
--------
Text chunking and embedding utilities for the Smart Notes Q&A Assistant.
"""

import re
from sentence_transformers import SentenceTransformer

# ── Model (loaded once at import time) ───────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Text chunking ─────────────────────────────────────────────────────────────

def split_into_chunks(text: str, min_words: int = 200, max_words: int = 300) -> list[str]:
    """
    Split *text* into overlapping word-level chunks.

    Strategy:
    1. Normalise whitespace and remove blank lines.
    2. Build chunks by accumulating sentences until we hit *max_words*.
    3. Each chunk targets 200–300 words for a good context window.

    Returns a list of non-empty text chunks.
    """
    # Normalise: collapse runs of whitespace / blank lines
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    # Split into sentences (rough heuristic – handles common abbreviations)
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)

    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        current_words.extend(words)

        if len(current_words) >= max_words:
            chunk = " ".join(current_words[:max_words])
            chunks.append(chunk)
            # Overlap: keep last 50 words from previous chunk for continuity
            current_words = current_words[max_words - 50:]

    # Flush remaining words if they meet the minimum size
    if current_words:
        if len(current_words) >= min_words or not chunks:
            chunks.append(" ".join(current_words))
        else:
            # Merge short tail into previous chunk
            if chunks:
                chunks[-1] = chunks[-1] + " " + " ".join(current_words)
            else:
                chunks.append(" ".join(current_words))

    return [c.strip() for c in chunks if c.strip()]


# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a list of float embeddings for the given texts."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Return a single embedding vector for a user query."""
    return embed_texts([query])[0]
