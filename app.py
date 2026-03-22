"""
app.py
------
Flask application for the Smart Notes Q&A Assistant (RAG using Endee).

Routes:
    GET  /          – Main UI (upload + question form)
    POST /upload    – Process uploaded text file → embed → store in Endee
    POST /ask       – Answer a question using RAG retrieval from Endee
    GET  /status    – JSON health check (Endee + model status)
"""

import os
import socket
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from utils import split_into_chunks, embed_texts
from endee_store import ensure_index, clear_index, store_chunks
from retrieve import answer_question

# ── App setup ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-in-prod")

ALLOWED_EXTENSIONS = {"txt", "md"}

# ── State (in-memory for this session) ────────────────────────────────────────
# Stores metadata about the currently loaded notes file
_notes_state: dict = {
    "filename": None,
    "num_chunks": 0,
    "ready": False,
}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _is_endee_reachable(timeout_seconds: float = 1.5) -> bool:
    """Fast socket-level check to avoid blocking on SDK calls when Endee is down."""
    host = "localhost"
    port = 8090
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", state=_notes_state)


@app.route("/upload", methods=["POST"])
def upload():
    """
    1. Accept a .txt / .md file upload.
    2. Read and chunk the text (200–300 words per chunk).
    3. Embed all chunks with sentence-transformers.
    4. Store embeddings in Endee (clears previous index first).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Only .txt and .md files are supported."}), 400

    filename = secure_filename(file.filename)
    raw_text = file.read().decode("utf-8", errors="replace")

    if not raw_text.strip():
        return jsonify({"error": "The uploaded file appears to be empty."}), 400

    try:
        # Step 1 – chunk the text
        chunks = split_into_chunks(raw_text)
        logger.info("Split '%s' into %d chunks.", filename, len(chunks))

        # Step 2 – embed all chunks
        embeddings = embed_texts(chunks)
        logger.info("Generated %d embeddings.", len(embeddings))

        # Step 3 – reset Endee index and store
        clear_index()
        num_stored = store_chunks(chunks, embeddings)

        # Update session state
        _notes_state["filename"] = filename
        _notes_state["num_chunks"] = num_stored
        _notes_state["ready"] = True

        return jsonify({
            "success": True,
            "filename": filename,
            "num_chunks": num_stored,
            "message": f"✓ Loaded {num_stored} chunks from '{filename}' into Endee.",
        })

    except Exception as exc:
        logger.exception("Upload failed: %s", exc)
        return jsonify({"error": f"Processing failed: {str(exc)}"}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    1. Accept a JSON or form-encoded question.
    2. Embed the question.
    3. Search Endee for the top-3 relevant chunks.
    4. Return a formatted answer.
    """
    if not _notes_state["ready"]:
        return jsonify({
            "error": "No notes loaded. Please upload a file first."
        }), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or request.form.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    use_llm = bool(data.get("use_llm", False))

    try:
        result = answer_question(question, top_k=3, use_llm=use_llm)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Ask failed: %s", exc)
        return jsonify({"error": f"Query failed: {str(exc)}"}), 500


@app.route("/status")
def status():
    """Quick health-check endpoint."""
    endee_ok = _is_endee_reachable()

    return jsonify({
        "endee": "ok" if endee_ok else "error – is Docker running?",
        "notes_loaded": _notes_state["ready"],
        "filename": _notes_state.get("filename"),
        "num_chunks": _notes_state.get("num_chunks", 0),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Skipping blocking Endee init at startup (lazy init on upload).")
    app.run(host="0.0.0.0", port=5000, debug=False)
