"""Central configuration. Values can be overridden with environment variables
(or a local .env file) so the same code runs locally and on Streamlit Cloud."""
from __future__ import annotations

import os
from pathlib import Path

try:  # use the operating system's certificate store (fixes SSL errors behind office firewalls)
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover
    pass

try:  # optional: load .env when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INTERNAL_DOCS_DIR = DATA_DIR / "raw" / "internal_docs"
REGULATIONS_DIR = DATA_DIR / "raw" / "regulations"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
REPORTS_DIR = ROOT / "reports"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
INDEX_META_PATH = PROCESSED_DIR / "index_meta.json"
EVAL_SET_PATH = EVAL_DIR / "eval_questions.csv"

# --- Gemini models (check https://ai.google.dev/gemini-api/docs/models for current names)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", CHAT_MODEL)

# --- Chunking
MAX_CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

# --- Retrieval
TOP_K = 5
RRF_K = 60  # reciprocal rank fusion constant
