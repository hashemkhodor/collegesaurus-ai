"""Shared configuration for the Collegesaurus AI chatbot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


REPO_ROOT = Path(__file__).resolve().parent

# Where Chroma persists vectors. Rebuilt by `python ingest.py`.
CHROMA_DIR = REPO_ROOT / "chroma_db"

# Cached source MDX files fetched from the Docusaurus repo.
SOURCE_CACHE_DIR = REPO_ROOT / "source_cache"

# Source repo — public, no auth needed.
SOURCE_REPO = os.environ.get("SOURCE_REPO", "hashemkhodor/collegesaurus")
SOURCE_BRANCH = os.environ.get("SOURCE_BRANCH", "main")

# Docusaurus routes. Used to generate citation URLs back to the live site.
SITE_BASE_URL = "https://hashemkhodor.github.io/collegesaurus"

# Directories in the source repo we ingest. Keep in sync with docusaurus.config.ts.
CONTENT_DIRS = ("universities", "scholarships")

# Files we skip (templates, intros).
SKIP_FILES = {"_template.mdx", "intro.mdx"}

# Chroma collection names (one per content type — lets the agent target either).
COLLECTIONS = {
    "universities": "universities",
    "scholarships": "scholarships",
}

# Gemini model names.
GEMINI_CHAT_MODEL = "gemini-2.5-flash-lite"
GEMINI_EMBED_MODEL = "gemini-embedding-001"

# Retrieval defaults.
# TOP_K bumped from 5 to 15: list-style questions (e.g. "engineering majors
# at AUB") need several chunks from the same table to return a complete
# answer — otherwise the model only sees the first ~3 rows.
TOP_K = 15
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
EMBED_BATCH_SIZE = 100  # items per embed_content API call

# Cap on the model's reply length (tokens). 10k leaves plenty of room for
# long majors tables or catalog listings; Flash-Lite's ceiling is 65,536.
MAX_OUTPUT_TOKENS = 10000


def require_api_key() -> str:
    """Return the Gemini API key or raise a helpful error."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://aistudio.google.com/app/apikey"
        )
    return key
