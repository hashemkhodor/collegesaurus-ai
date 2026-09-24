"""Settings for the chatbot service, read from the environment (and .env)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIR = PACKAGE_DIR / "web"

# The live site publishes its content for us (see chatbot/README.md).
SITE_URL = os.environ.get("SITE_URL", "https://collegesaurus.org").rstrip("/")
CORPUS_URLS = {
    "en": f"{SITE_URL}/chatbot/corpus.json",
    "ar": f"{SITE_URL}/ar/chatbot/corpus.json",
}
VERSION_URL = f"{SITE_URL}/chatbot/version.json"

# Snapshot baked into the image by the deploy workflow; loaded at boot.
INDEX_PATH = Path(os.environ.get("INDEX_PATH", PACKAGE_DIR.parent / "data" / "index.db"))

GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768
TOP_K = 15
MAX_STEPS = 6  # model calls per answer, tool calls included
MAX_OUTPUT_TOKENS = 8192

POLL_SECONDS = 60  # how often to check the site's version.json
STALE_AFTER_SECONDS = 600  # /healthz reports stale after this long without a good poll
KEEPALIVE_SECONDS = 4 * 3600  # keeps the free Supabase project from pausing

# Request limits.
MAX_USER_CHARS = 1000
MAX_ASSISTANT_CHARS = 4000  # longer assistant turns in the history are truncated
MAX_HISTORY_MESSAGES = 20
MAX_BODY_BYTES = 32 * 1024
SESSION_LIMIT = (10, 30)  # turns per seconds, per browser session
IP_LIMIT = (30, 60)  # turns per seconds, per client IP (shared IPs are common)
DAILY_LIMIT = int(os.environ.get("DAILY_TURN_LIMIT", "1500"))  # all users, per UTC day

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")
IP_HASH_SECRET = os.environ.get("IP_HASH_SECRET", "")

# Pages allowed to embed the chat in an iframe.
FRAME_ANCESTORS = os.environ.get(
    "FRAME_ANCESTORS", "https://collegesaurus.org http://localhost:3000"
).split()


def gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/app/apikey"
        )
    return key
