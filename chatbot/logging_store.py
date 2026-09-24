"""Chat logs and feedback, written to Supabase over its REST API.

Fire-and-forget: a failed write is printed to stderr and dropped, so chat
keeps working when Supabase is down or not configured. The schema lives in
supabase/chat_logs.sql. Raw IP addresses are never stored; `ip_hmac` gives a
keyed hash that can link abuse from one address without revealing it.
"""

from __future__ import annotations

import hashlib
import hmac
import sys

import httpx


class ChatLogger:
    def __init__(
        self, url: str, key: str, ip_secret: str = "", *, client: httpx.AsyncClient | None = None
    ):
        self._url = url.rstrip("/")
        self._key = key
        self._ip_secret = ip_secret.encode()
        self._client = client or httpx.AsyncClient(timeout=10.0)

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._key)

    def ip_hmac(self, ip: str) -> str | None:
        if not (ip and self._ip_secret):
            return None
        return hmac.new(self._ip_secret, ip.encode(), hashlib.sha256).hexdigest()[:32]

    async def log_turn(self, row: dict) -> None:
        await self._send("POST", "chat_logs", json=row)

    async def log_feedback(self, turn_id: str, value: int) -> None:
        await self._send("POST", "chat_feedback", json={"turn_id": turn_id, "value": value})

    async def keepalive(self) -> None:
        """A tiny read, so a free-plan project doesn't pause during quiet weeks."""
        await self._send("GET", "chat_logs", params={"select": "turn_id", "limit": "1"})

    async def _send(self, method: str, table: str, **kwargs) -> None:
        if not self.enabled:
            return
        headers = {"apikey": self._key, "Prefer": "return=minimal"}
        if self._key.startswith("eyJ"):
            # Legacy anon/service_role keys are JWTs; new sb_secret_ keys are not
            # and belong only in the apikey header.
            headers["Authorization"] = f"Bearer {self._key}"
        try:
            response = await self._client.request(
                method, f"{self._url}/rest/v1/{table}", headers=headers, **kwargs
            )
            if response.status_code >= 300:
                print(
                    f"[logging_store] {method} {table}: HTTP {response.status_code} "
                    f"{response.text[:200]}",
                    file=sys.stderr,
                )
        except httpx.HTTPError as exc:
            print(f"[logging_store] {method} {table}: {exc}", file=sys.stderr)
