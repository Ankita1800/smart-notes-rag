"""
endee_store.py
--------------
All Endee vector database operations for the Smart Notes Q&A Assistant.

Endee is a high-performance open-source vector database.
It runs as a local server on port 8080 (via Docker).

Setup:
    docker compose up -d          # starts endee-server on localhost:8080

Python SDK:
    pip install endee

Docs: https://docs.endee.io
"""

import uuid
import logging
from endee import Endee, Precision
from utils import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_NAME = "notes_index"
ENDEE_URL = "http://localhost:8090/api/v1"
# If you set NDD_AUTH_TOKEN on the server, pass the same value here:
ENDEE_AUTH_TOKEN = ""   # leave empty for local dev (no auth)


# ── Client factory ────────────────────────────────────────────────────────────

def _get_client() -> Endee:
    """Return a configured Endee client."""
    client = Endee(ENDEE_AUTH_TOKEN if ENDEE_AUTH_TOKEN else None)
    client.set_base_url(ENDEE_URL)
    return client


# ── Index lifecycle ───────────────────────────────────────────────────────────

def ensure_index() -> None:
    """
    Create the notes index if it doesn't already exist.

    Index parameters:
    - dimension  : 384  (all-MiniLM-L6-v2 output size)
    - space_type : cosine  (normalised semantic similarity)
    - precision  : INT8    (fast, memory-efficient quantisation)
    """
    client = _get_client()
    try:
        client.get_index(name=INDEX_NAME)
        logger.info("Endee index '%s' already exists.", INDEX_NAME)
    except Exception:
        logger.info("Creating Endee index '%s' …", INDEX_NAME)
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
        logger.info("Index '%s' created successfully.", INDEX_NAME)


def clear_index() -> None:
    """
    Drop and re-create the notes index.
    Called each time the user uploads a new file so the index is fresh.
    """
    client = _get_client()
    try:
        client.delete_index(name=INDEX_NAME)
        logger.info("Dropped existing index '%s'.", INDEX_NAME)
    except Exception:
        pass  # index may not exist yet
    ensure_index()


# ── Vector storage ────────────────────────────────────────────────────────────

def store_chunks(chunks: list[str], embeddings: list[list[float]]) -> int:
    """
    Upsert text chunks and their embeddings into Endee.

    Each vector document has:
      id     : unique UUID string
      vector : 384-dimensional float list
      meta   : {"text": <original chunk text>, "chunk_index": <int>}

    Returns the number of vectors stored.
    """
    client = _get_client()
    index = client.get_index(name=INDEX_NAME)

    documents = [
        {
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "meta": {
                "text": chunk,
                "chunk_index": i,
            },
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    index.upsert(documents)
    logger.info("Stored %d chunks in Endee index '%s'.", len(documents), INDEX_NAME)
    return len(documents)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search_chunks(query_embedding: list[float], top_k: int = 3) -> list[dict]:
    """
    Search Endee for the *top_k* most similar chunks.

    Returns a list of dicts:
        [{"text": str, "chunk_index": int, "similarity": float}, ...]
    """
    client = _get_client()
    index = client.get_index(name=INDEX_NAME)

    results = index.query(vector=query_embedding, top_k=top_k)

    hits = []
    for result in results:
        if isinstance(result, dict):
            meta = result.get("meta") or result.get("metadata") or {}
            similarity = result.get("similarity")
            if similarity is None:
                similarity = result.get("score", 0.0)
        else:
            meta = getattr(result, "meta", None) or getattr(result, "metadata", None) or {}
            similarity = getattr(result, "similarity", None)
            if similarity is None:
                similarity = getattr(result, "score", 0.0)

        hits.append({
            "text": meta.get("text", ""),
            "chunk_index": meta.get("chunk_index", -1),
            "similarity": round(float(similarity or 0.0), 4),
        })

    return hits
