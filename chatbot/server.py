"""The FastAPI service: the chat page, the streaming chat API, feedback and health.

    uvicorn chatbot.server:app --reload          # locally; needs GEMINI_API_KEY

At startup it loads the index snapshot baked into the image (config.INDEX_PATH),
then checks the live site every config.POLL_SECONDS and swaps in a fresh index
when the content changed. It runs as a single always-on instance, so rate
limits are kept in memory.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, Field, ValidationError

from chatbot import config
from chatbot.agent import ChatTurn, system_prompt
from chatbot.ingest import Embedder, GeminiEmbedder, Refresher
from chatbot.logging_store import ChatLogger
from chatbot.sources import CollegesaurusCorpus
from chatbot.store import Page, SqliteNumpyStore
from chatbot.tools import ToolRunner, question_locale

# Messages the server itself shows. The UI's own labels live in web/chat.js.
MESSAGES = {
    "en": {
        "out_of_scope": (
            "I can only help with Lebanese universities and the scholarships on "
            "Collegesaurus. Try asking about a major, a university or a scholarship."
        ),
        "busy": "The model is busy right now — please try again in a few seconds.",
        "error": "Something went wrong. Please try again.",
        "rate_limited": "Too many questions too fast — please wait a moment and try again.",
        "daily_cap": "We've reached today's question limit — please come back tomorrow.",
        "warming_up": "The assistant is starting up — please try again in a minute.",
        "too_long": "That question is too long — please keep it under {max} characters.",
        "too_large": "This conversation is too long — please start a new chat.",
        "invalid_request": "Something went wrong — please reload the page.",
    },
    "ar": {
        "out_of_scope": (
            "أستطيع المساعدة فقط في الجامعات اللبنانية والمنح الموجودة على كوليجسورس. "
            "اسأل عن تخصّص أو جامعة أو منحة."
        ),
        "busy": "النموذج مشغول حاليًا — يُرجى المحاولة بعد لحظات.",
        "error": "حدث خطأ ما. يُرجى المحاولة مرة أخرى.",
        "rate_limited": "أسئلة كثيرة خلال وقت قصير — انتظر قليلًا ثم حاول مجددًا.",
        "daily_cap": "وصلنا إلى الحد اليومي للأسئلة — عُد غدًا من فضلك.",
        "warming_up": "المساعد قيد التشغيل — حاول بعد دقيقة.",
        "too_long": "السؤال طويل جدًا — أبقه ضمن {max} حرفًا.",
        "too_large": "المحادثة طويلة جدًا — ابدأ محادثة جديدة.",
        "invalid_request": "حدث خطأ ما — أعد تحميل الصفحة.",
    },
    "fr": {
        "out_of_scope": (
            "Je ne peux répondre qu'aux questions sur les universités libanaises et les "
            "bourses présentes sur Collegesaurus. Posez une question sur une spécialité, "
            "une université ou une bourse."
        ),
        "busy": "Le modèle est très sollicité — réessayez dans quelques secondes.",
        "error": "Une erreur s'est produite. Veuillez réessayer.",
        "rate_limited": "Trop de questions trop vite — patientez un instant puis réessayez.",
        "daily_cap": "La limite de questions du jour est atteinte — revenez demain.",
        "warming_up": "L'assistant démarre — réessayez dans une minute.",
        "too_long": "Question trop longue — restez sous {max} caractères.",
        "too_large": "Cette conversation est trop longue — commencez-en une nouvelle.",
        "invalid_request": "Une erreur s'est produite — rechargez la page.",
    },
}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    lang: str = "en"  # UI language: en | ar | fr
    page: str | None = Field(default=None, max_length=300)  # site path the chat was opened on
    session_id: str = Field(default="", max_length=64)


class Feedback(BaseModel):
    turn_id: uuid.UUID
    value: Literal[1, -1]


@dataclass(frozen=True)
class Limits:
    session: tuple[int, float] = config.SESSION_LIMIT
    ip: tuple[int, float] = config.IP_LIMIT
    daily: int = config.DAILY_LIMIT


class RateLimiter:
    """Sliding-window counters, in memory (the app runs as one instance)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, rules: list[tuple[str, str, int, float]]) -> str | None:
        """`rules` are (name, key, limit, window seconds). Returns the name of the
        first rule already at its limit; otherwise records a hit on every rule."""
        now = self._clock()
        for name, key, limit, window in rules:
            hits = self._hits.get(key)
            while hits and now - hits[0] >= window:
                hits.popleft()
            if hits is not None and len(hits) >= limit:
                return name
        for _, key, _, _ in rules:
            self._hits.setdefault(key, deque()).append(now)
        if len(self._hits) > 50_000:
            for key in [k for k, q in self._hits.items() if not q or now - q[-1] > 86_400]:
                del self._hits[key]
        return None


@dataclass
class Services:
    refresher: Refresher
    embedder: Embedder
    genai_client: object  # google.genai.Client
    logger: ChatLogger
    page_types: dict[str, str]
    limiter: RateLimiter = field(default_factory=RateLimiter)
    limits: Limits = field(default_factory=Limits)
    model: str = config.GEMINI_CHAT_MODEL


@dataclass(frozen=True)
class _Turn:
    turn_id: str
    session_id: str
    lang: str
    page: str | None
    question: str
    ip_hmac: str | None
    index_version: str | None


def create_app(
    services: Services | None = None, *, web_dir: Path = config.WEB_DIR, background: bool = True
) -> FastAPI:
    """`services` defaults to the production wiring, built at startup from the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.services is None:
            app.state.services = _production_services()
        loops = []
        if background:
            s = app.state.services
            loops = [
                asyncio.create_task(_poll(s.refresher)),
                asyncio.create_task(_keepalive(s.logger)),
            ]
        yield
        for loop in loops:
            loop.cancel()

    app = FastAPI(
        title="Collegesaurus AI", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.services = services
    csp = "; ".join(
        [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self'",
            "img-src 'self' data:",
            "connect-src 'self'",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors " + " ".join(["'self'", *config.FRAME_ANCESTORS]),
        ]
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            # Revalidate (cheap, ETags) so a deploy never leaves browsers on old JS/CSS.
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return _error(400, "invalid_request", "en")

    # HEAD too: uptime monitors often probe with it.
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def index():
        return FileResponse(web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=web_dir, check_dir=False), name="static")

    @app.api_route("/healthz", methods=["GET", "HEAD"])
    async def healthz():
        refresher = app.state.services.refresher
        store = refresher.store
        body = {
            "ok": store is not None,
            "stale": refresher.stale,
            "last_ok_at": datetime.fromtimestamp(refresher.last_ok_at, UTC).isoformat(
                timespec="seconds"
            ),
            "last_error": refresher.last_error,
            "index": {
                k: store.meta.get(k)
                for k in ("build_id", "content_sha", "chunks", "built_at", "model")
            }
            if store
            else None,
        }
        return JSONResponse(body, status_code=200 if store else 503)

    @app.post("/api/chat")
    async def chat(request: Request):
        s: Services = app.state.services
        raw = await _read_limited(request, config.MAX_BODY_BYTES)
        if raw is None:
            return _error(413, "too_large", "en")
        try:
            body = ChatRequest.model_validate_json(raw)
        except ValidationError:
            return _error(400, "invalid_request", "en")
        last = body.messages[-1]
        question = last.content.strip()
        lang = _message_lang(body.lang, question)
        if last.role != "user" or not question:
            return _error(400, "invalid_request", lang)
        if len(question) > config.MAX_USER_CHARS:
            return _error(400, "too_long", lang)
        store = s.refresher.store
        if store is None:
            return _error(503, "warming_up", lang)
        ip = _client_ip(request)
        rules = [
            ("rate_limited", f"ip:{ip}", *s.limits.ip),
            ("daily_cap", "day", s.limits.daily, 86_400.0),
        ]
        if body.session_id:
            rules.insert(0, ("rate_limited", f"session:{body.session_id}", *s.limits.session))
        exceeded = s.limiter.check(rules)
        if exceeded:
            return _error(429, exceeded, lang)

        runner = ToolRunner(
            store, s.embedder.embed_query, s.page_types, locale=question_locale(question)
        )
        turn = ChatTurn(
            s.genai_client,
            runner,
            system_instruction=system_prompt(s.page_types, _page_for(store, body.page)),
            model=s.model,
        )
        context = _Turn(
            turn_id=str(uuid.uuid4()),
            session_id=body.session_id,
            lang=lang,
            page=body.page,
            question=question,
            ip_hmac=s.logger.ip_hmac(ip),
            index_version=store.meta.get("content_sha"),
        )
        return StreamingResponse(
            _stream(s, turn, _history(body.messages), context),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/feedback")
    async def feedback(body: Feedback, request: Request):
        s: Services = app.state.services
        if s.limiter.check([("rate_limited", f"feedback:{_client_ip(request)}", 30, 60.0)]):
            return _error(429, "rate_limited", "en")
        await s.logger.log_feedback(str(body.turn_id), body.value)
        return Response(status_code=204)

    return app


async def _stream(s: Services, turn: ChatTurn, history, ctx: _Turn) -> AsyncIterator[str]:
    finished = False
    try:
        yield _sse("meta", {"turn_id": ctx.turn_id})
        async for kind, data in turn.stream(history):
            yield _sse(kind, data)
        outcome = turn.result.outcome
        if outcome == "out_of_scope":
            yield _sse("delta", {"text": _message(ctx.lang, "out_of_scope")})
        elif outcome in ("busy", "error"):
            yield _sse("error", {"code": outcome, "message": _message(ctx.lang, outcome)})
        yield _sse("done", {"turn_id": ctx.turn_id, "outcome": outcome})
        finished = True
        await s.logger.log_turn(_log_row(s, turn, ctx))
    finally:
        if not finished:  # the client left mid-answer; log it without blocking
            _in_background(s.logger.log_turn(_log_row(s, turn, ctx)))


def _log_row(s: Services, turn: ChatTurn, ctx: _Turn) -> dict:
    r = turn.result
    shown = _message(ctx.lang, r.outcome) if r.outcome in ("out_of_scope", "busy", "error") else ""
    return {
        "turn_id": ctx.turn_id,
        "session_id": ctx.session_id or None,
        "lang": ctx.lang,
        "page": ctx.page,
        "question": ctx.question,
        "answer": r.answer or shown,
        "outcome": r.outcome,
        "tool_calls": r.tool_calls,
        "sources": r.sources,
        "top_score": r.top_score,
        "error": r.error,
        "latency_ms": r.latency_ms,
        "model": s.model,
        "index_version": ctx.index_version,
        "ip_hmac": ctx.ip_hmac,
    }


def _history(messages: list[Message]) -> list[tuple[str, str]]:
    """Recent turns only, each trimmed, starting with the user, roles alternating."""
    turns: list[tuple[str, str]] = []
    for m in messages[-config.MAX_HISTORY_MESSAGES :]:
        limit = config.MAX_USER_CHARS if m.role == "user" else config.MAX_ASSISTANT_CHARS
        text = m.content.strip()[:limit]
        if not text:
            continue
        if turns and turns[-1][0] == m.role:
            turns[-1] = (m.role, f"{turns[-1][1]}\n\n{text}")
        elif turns or m.role == "user":
            turns.append((m.role, text))
    return turns


def _page_for(store: SqliteNumpyStore, path: str | None) -> Page | None:
    """The indexed page at site path `path` (e.g. "/ar/universities/aub"), if any."""
    if not path:
        return None
    wanted = path.split("?")[0].split("#")[0].rstrip("/")
    for chunk in store.chunks:
        if urlparse(chunk.url).path.rstrip("/") == wanted:
            return Page(chunk.doc_id, chunk.type, chunk.title, chunk.url)
    return None


async def _read_limited(request: Request, limit: int) -> bytes | None:
    """The request body, or None if it is larger than `limit` bytes."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    body = b""
    async for piece in request.stream():
        body += piece
        if len(body) > limit:
            return None
    return body


def _client_ip(request: Request) -> str:
    # Fly.io's proxy sets Fly-Client-IP to the real client address.
    return request.headers.get("fly-client-ip") or (request.client.host if request.client else "")


def _message_lang(requested: str, question: str) -> str:
    if question and question_locale(question) == "ar":
        return "ar"
    return requested if requested in MESSAGES else "en"


def _message(lang: str, code: str) -> str:
    return MESSAGES.get(lang, MESSAGES["en"])[code].format(max=config.MAX_USER_CHARS)


def _error(status: int, code: str, lang: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": _message(lang, code)}, status_code=status)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_BACKGROUND: set[asyncio.Task] = set()


def _in_background(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


async def _poll(refresher: Refresher) -> None:
    while True:
        if await asyncio.to_thread(refresher.refresh_once):
            meta = refresher.store.meta
            print(f"[index] serving {meta['build_id']} ({meta['chunks']} chunks)", flush=True)
        elif refresher.last_error:
            print(f"[index] refresh failed: {refresher.last_error}", file=sys.stderr, flush=True)
        await asyncio.sleep(config.POLL_SECONDS)


async def _keepalive(logger: ChatLogger) -> None:
    while True:
        await logger.keepalive()
        await asyncio.sleep(config.KEEPALIVE_SECONDS)


def _production_services() -> Services:
    client = genai.Client(api_key=config.gemini_api_key())
    embedder = GeminiEmbedder(client)
    store = SqliteNumpyStore.load(config.INDEX_PATH) if config.INDEX_PATH.exists() else None
    sources = [CollegesaurusCorpus(config.CORPUS_URLS, version=config.VERSION_URL)]
    return Services(
        refresher=Refresher(sources, embedder, store),
        embedder=embedder,
        genai_client=client,
        logger=ChatLogger(config.SUPABASE_URL, config.SUPABASE_SECRET_KEY, config.IP_HASH_SECRET),
        page_types={kind: about for source in sources for kind, about in source.types.items()},
    )


app = create_app()
