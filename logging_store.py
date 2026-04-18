"""Append-only log of chat turns to Supabase Postgres.

Fire-and-forget: a failed insert logs to stderr and returns. Chat stays
responsive even if Supabase is down or misconfigured. If SUPABASE_URL is
empty, `log_turn` is a no-op so local dev without a Supabase project works
the same as today.
"""

from __future__ import annotations

import sys
from typing import Optional

import config

_client = None
_init_failed = False


def _get_client():
    global _client, _init_failed
    if _client is not None or _init_failed:
        return _client
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        _init_failed = True
        return None
    try:
        from supabase import create_client

        _client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    except Exception as exc:
        print(f"[logging_store] supabase client init failed: {exc}", file=sys.stderr)
        _init_failed = True
    return _client


def log_turn(
    session_id: str,
    lang: str,
    question: str,
    answer: str,
    tool_calls: list[str],
    latency_ms: int,
    error: Optional[str] = None,
) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.table(config.LOG_TABLE).insert(
            {
                "session_id": session_id,
                "lang": lang,
                "question": question,
                "answer": answer,
                "tool_calls": tool_calls,
                "latency_ms": latency_ms,
                "error": error,
            }
        ).execute()
    except Exception as exc:
        print(f"[logging_store] insert failed: {exc}", file=sys.stderr)
