"""Central configuration for Nyaya-Sahayak v2.

Reads env vars at import time. All knobs in one file.
"""
import os

# ---------------------------------------------------------------------------
# LLM — Sarvam AI (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
LLM_OPENAI_BASE_URL = os.environ.get(
    "LLM_OPENAI_BASE_URL",
    "https://api.sarvam.ai/v1",
).strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "sarvam-m").strip()
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# Embeddings (CPU-friendly, 384-dim)
# ---------------------------------------------------------------------------
DENSE_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# FAISS index (built from Delta table, stored in UC Volume)
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = os.environ.get(
    "FAISS_INDEX_DIR",
    "/Volumes/workspace/default/bharat_bricks_hacks/nyaya_index",
).strip()
LOCAL_INDEX_CACHE = "/tmp/nyaya_index"

# ---------------------------------------------------------------------------
# Unity Catalog
# ---------------------------------------------------------------------------
CATALOG = os.environ.get("NYAYA_CATALOG", "workspace")
SCHEMA = os.environ.get("NYAYA_SCHEMA", "default")
TABLE = os.environ.get("NYAYA_TABLE", "legal_rag_corpus")
VOLUME = os.environ.get("NYAYA_VOLUME", "bharat_bricks_hacks")
VOL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# ---------------------------------------------------------------------------
# Agent graph
# ---------------------------------------------------------------------------
MAX_TOOL_CALLS = int(os.environ.get("AGENT_MAX_TOOL_CALLS", "6"))
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "8"))
GRAPH_RECURSION_LIMIT = int(os.environ.get("AGENT_RECURSION_LIMIT", "40"))
BASE_TOKEN_THRESHOLD = int(os.environ.get("AGENT_TOKEN_THRESHOLD", "2000"))
TOKEN_GROWTH_FACTOR = float(os.environ.get("AGENT_TOKEN_GROWTH", "0.9"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
SEARCH_K = int(os.environ.get("SEARCH_K", "10"))
CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE", "500"))
CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP", "100"))
MIN_PARENT_SIZE = int(os.environ.get("MIN_PARENT_SIZE", "2000"))
MAX_PARENT_SIZE = int(os.environ.get("MAX_PARENT_SIZE", "4000"))

# ---------------------------------------------------------------------------
# Document Processing (PDF → Markdown → Chunks)
# ---------------------------------------------------------------------------
MARKDOWN_DIR = os.environ.get(
    "MARKDOWN_DIR", f"{VOL_PATH}/markdown_docs"
).strip()
PDF_DIR = os.environ.get(
    "PDF_DIR", f"{VOL_PATH}/pdfs"
).strip()
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

# ---------------------------------------------------------------------------
# Sarvam AI (voice / translation)
# ---------------------------------------------------------------------------
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Sarvam Languages
# ---------------------------------------------------------------------------
SARVAM_LANGUAGES: list[tuple[str, str]] = [
    ("en", "English"),
    ("hi", "Hindi · हिन्दी"),
    ("bn", "Bengali"),
    ("te", "Telugu"),
    ("mr", "Marathi"),
    ("ta", "Tamil"),
    ("gu", "Gujarati"),
    ("kn", "Kannada"),
    ("ml", "Malayalam"),
    ("pa", "Punjabi"),
    ("or", "Odia"),
    ("ur", "Urdu"),
    ("as", "Assamese"),
]

UI_TO_BCP47: dict[str, str] = {
    "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN", "te": "te-IN",
    "mr": "mr-IN", "ta": "ta-IN", "gu": "gu-IN", "kn": "kn-IN",
    "ml": "ml-IN", "pa": "pa-IN", "or": "od-IN", "ur": "hi-IN",
    "as": "bn-IN",
}
