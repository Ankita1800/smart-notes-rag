# Smart Notes Q&A Assistant
### RAG (Retrieval-Augmented Generation) using Endee Vector Database

---

## Project Overview

A **2-day student project** that turns any plain-text notes file into an intelligent Q&A assistant.  
Upload your notes → ask questions → get back the most relevant sections, powered entirely by local semantic search.  
**No API key required. No internet calls. Runs 100% on your machine.**

---

## Problem Statement

Students accumulate large text files of notes (lectures, textbooks, revision guides) but struggle to find specific information quickly. Simple keyword search misses paraphrased content. This project solves that by converting notes into semantic vector embeddings and retrieving the most *meaning-relevant* sections for any question.

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a two-step AI pattern:

```
1. INDEXING   → Split documents into chunks → Embed each chunk as a vector → Store in vector DB
2. RETRIEVAL  → Embed the user question → Find closest vectors → Return relevant chunks
```

Unlike a general LLM, RAG grounds answers in *your* documents — making it accurate, controllable, and explainable.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER BROWSER                        │
│         Upload .txt file     /     Ask question         │
└────────────────┬────────────────────────┬───────────────┘
                 │ POST /upload           │ POST /ask
                 ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│                    FLASK  (app.py)                      │
└──────┬──────────────────────────────────────┬───────────┘
       │                                      │
       ▼                                      ▼
┌─────────────┐                      ┌──────────────────┐
│  utils.py   │                      │   retrieve.py    │
│  • chunk    │                      │  • embed query   │
│  • embed    │                      │  • search Endee  │
└──────┬──────┘                      └────────┬─────────┘
       │                                      │
       ▼                                      ▼
┌──────────────────────────────────────────────────────────┐
│                 endee_store.py                           │
│   create_index / upsert vectors / query top-k           │
└──────────────────────────┬───────────────────────────────┘
                           │  HTTP  localhost:8090
                           ▼
┌──────────────────────────────────────────────────────────┐
│           ENDEE VECTOR DATABASE  (Docker)               │
│   HNSW index · cosine similarity · INT8 quantisation    │
└──────────────────────────────────────────────────────────┘
```

---

## File Structure

```
smart-notes-rag/
├── app.py              ← Flask routes (upload, ask, status)
├── endee_store.py      ← All Endee vector DB operations
├── retrieve.py         ← RAG pipeline (embed → search → format)
├── utils.py            ← Text chunking + embedding helpers
├── llm_optional.py     ← Optional LLM enhancement (OpenAI)
├── requirements.txt    ← Python dependencies
├── docker-compose.yml  ← Endee server setup
├── README.md
└── templates/
    └── index.html      ← Frontend UI
```

---

## Endee Usage (Detailed)

**Endee** is a high-performance open-source vector database that runs as a local server.  
Docs: https://docs.endee.io

### Why Endee?
- Runs locally via Docker — no cloud account, no API key
- HNSW (Hierarchical Navigable Small World) algorithm for fast ANN search
- Supports cosine / L2 / inner-product distance metrics
- INT8 quantisation: fast + memory-efficient
- Clean Python SDK: `pip install endee`

### How this project uses Endee

| Operation | Code location | Endee call |
|-----------|--------------|------------|
| Create index (dim=384, cosine, INT8) | `endee_store.ensure_index()` | `client.create_index(...)` |
| Drop + recreate on new upload | `endee_store.clear_index()` | `client.delete_index(...)` |
| Store chunk + embedding + metadata | `endee_store.store_chunks()` | `index.upsert([...])` |
| Semantic search — top 3 results | `endee_store.search_chunks()` | `index.query(vector, top_k=3)` |

### Index parameters explained

```python
client.create_index(
    name       = "notes_index",
    dimension  = 384,          # all-MiniLM-L6-v2 output size
    space_type = "cosine",     # normalised dot product — best for semantic search
    precision  = Precision.INT8  # 4× memory reduction vs FLOAT32, minimal accuracy loss
)
```

### Vector document schema

```python
{
    "id":     "550e8400-e29b-41d4-a716-446655440000",  # UUID
    "vector": [0.023, -0.14, ..., 0.089],              # 384 floats
    "meta":   {
        "text":        "The French Revolution began in 1789 ...",
        "chunk_index": 4
    }
}
```

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker + Docker Compose v2

### Step 1 — Start Endee (vector database)

```bash
docker compose up -d
```

Endee now runs at **http://localhost:8090**.  
Dashboard: http://localhost:8090 (view your indexes visually)

### Step 2 — Python environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> First run downloads `all-MiniLM-L6-v2` (~90 MB) automatically.

### Step 3 — Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Running WITHOUT an API Key (Default)

This is the **default mode** — no configuration needed.

1. Start Endee: `docker compose up -d`
2. Start Flask: `python app.py`
3. Upload a `.txt` file → ask questions

The system uses `sentence-transformers/all-MiniLM-L6-v2` locally for embeddings and returns formatted chunks directly from Endee — **zero external API calls**.

---

## Optional: LLM Enhancement

To generate a fluent synthesised answer (instead of raw chunks), you can optionally wire in OpenAI:

```bash
pip install openai
export OPENAI_API_KEY=sk-your-key-here
```

Then pass `use_llm: true` in the `/ask` POST body, or modify the frontend JS:

```js
body: JSON.stringify({ question: q, use_llm: true })
```

The LLM is **completely optional** — if the key is missing or the call fails, the system falls back to raw retrieval automatically. See `llm_optional.py`.

---

## API Reference

### `POST /upload`

| Field | Type | Description |
|-------|------|-------------|
| `file` | multipart | `.txt` or `.md` notes file |

**Response:**
```json
{
  "success": true,
  "filename": "notes.txt",
  "num_chunks": 12,
  "message": "✓ Loaded 12 chunks from 'notes.txt' into Endee."
}
```

### `POST /ask`

```json
{ "question": "What caused the French Revolution?", "use_llm": false }
```

**Response:**
```json
{
  "question": "What caused the French Revolution?",
  "answer": "Based on your notes, here are the most relevant sections...",
  "sources": [
    { "text": "...", "chunk_index": 4, "similarity": 0.8821 }
  ],
  "llm_used": false
}
```

### `GET /status`

Returns Endee connection status and notes load state.

---

## Example Queries

Assuming you upload lecture notes on Computer Science:

```
Q: What is Big O notation?
Q: Explain the difference between a stack and a queue
Q: How does quicksort work?
Q: What are the SOLID principles?
Q: What is a binary search tree?
```

Assuming history notes:

```
Q: What were the causes of World War 1?
Q: Who was Napoleon Bonaparte?
Q: Explain the Industrial Revolution
```

---

## Tech Stack

| Component | Library / Tool | Purpose |
|-----------|---------------|---------|
| Web framework | Flask 3 | HTTP server and routing |
| Embeddings | sentence-transformers | Local vector generation |
| Embedding model | all-MiniLM-L6-v2 | 384-dim, fast, high quality |
| Vector DB | Endee (Docker) | Store and search embeddings |
| Vector DB SDK | endee (pip) | Python client for Endee |
| Frontend | Vanilla HTML/CSS/JS | Zero-framework clean UI |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused localhost:8090` | Run `docker compose up -d` |
| `No module named 'endee'` | Run `pip install endee` |
| `No module named 'sentence_transformers'` | Run `pip install sentence-transformers` |
| Slow first query | Model is loading (~3–5s first time, cached after) |
| Empty answer | Upload a notes file first |

---

## License

MIT — free to use, modify, and learn from.
