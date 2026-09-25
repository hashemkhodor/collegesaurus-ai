import json

import httpx
import pytest

from chatbot.logging_store import ChatLogger

SUPABASE = "https://project.supabase.co"
SECRET_KEY = "sb_secret_abc123"
LEGACY_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.sig"
ROW = {
    "session_id": "s1",
    "turn_id": "7f1d2c9e-0000-4000-8000-000000000001",
    "lang": "en",
    "question": "AUB tuition?",
    "answer": "AUB charges $1,000 per credit.",
    "outcome": "answered",
    "sources": [{"title": "AUB", "url": "https://collegesaurus.org/universities/aub"}],
    "top_score": 0.81,
}


def recording_logger(key=SECRET_KEY, status=200, url=SUPABASE, ip_secret="pepper"):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=[] if request.method == "GET" else None)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ChatLogger(url, key, ip_secret, client=client), seen


@pytest.mark.anyio
async def test_log_turn_inserts_the_row_into_chat_logs():
    logger, seen = recording_logger()

    await logger.log_turn(ROW)

    [request] = seen
    assert (request.method, str(request.url)) == ("POST", f"{SUPABASE}/rest/v1/chat_logs")
    assert json.loads(request.content) == ROW
    assert request.headers["apikey"] == SECRET_KEY
    assert request.headers["prefer"] == "return=minimal"


@pytest.mark.anyio
async def test_new_secret_keys_go_only_in_the_apikey_header():
    logger, seen = recording_logger(key=SECRET_KEY)

    await logger.log_turn(ROW)

    assert "authorization" not in seen[0].headers


@pytest.mark.anyio
async def test_legacy_jwt_keys_are_also_sent_as_a_bearer_token():
    logger, seen = recording_logger(key=LEGACY_KEY)

    await logger.log_turn(ROW)

    assert seen[0].headers["authorization"] == f"Bearer {LEGACY_KEY}"


@pytest.mark.anyio
async def test_without_supabase_settings_nothing_is_sent():
    logger, seen = recording_logger(url="")

    await logger.log_turn(ROW)
    await logger.log_feedback(ROW["turn_id"], 1)
    await logger.keepalive()

    assert logger.enabled is False
    assert seen == []


@pytest.mark.anyio
async def test_a_failed_insert_is_reported_but_never_raised(capsys):
    logger, _ = recording_logger(status=500)

    await logger.log_turn(ROW)

    assert "chat_logs" in capsys.readouterr().err


@pytest.mark.anyio
async def test_an_unreachable_supabase_is_reported_but_never_raised(capsys):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    logger = ChatLogger(SUPABASE, SECRET_KEY, "pepper", client=client)

    await logger.log_turn(ROW)

    assert "connection refused" in capsys.readouterr().err


@pytest.mark.anyio
async def test_feedback_is_inserted_as_its_own_row():
    logger, seen = recording_logger()

    await logger.log_feedback(ROW["turn_id"], -1)

    [request] = seen
    assert str(request.url) == f"{SUPABASE}/rest/v1/chat_feedback"
    assert json.loads(request.content) == {"turn_id": ROW["turn_id"], "value": -1}


@pytest.mark.anyio
async def test_keepalive_reads_one_row():
    logger, seen = recording_logger()

    await logger.keepalive()

    [request] = seen
    assert request.method == "GET"
    assert request.url.path == "/rest/v1/chat_logs"
    assert dict(request.url.params) == {"select": "turn_id", "limit": "1"}


def test_ip_hmac_is_stable_per_ip_and_never_the_raw_address():
    logger, _ = recording_logger(ip_secret="pepper")

    first = logger.ip_hmac("203.0.113.7")

    assert first == logger.ip_hmac("203.0.113.7")
    assert first != logger.ip_hmac("203.0.113.8")
    assert "203.0.113.7" not in first
    assert first != recording_logger(ip_secret="salt")[0].ip_hmac("203.0.113.7")


def test_ip_hmac_is_empty_without_a_secret_or_an_ip():
    assert recording_logger(ip_secret="")[0].ip_hmac("203.0.113.7") is None
    assert recording_logger()[0].ip_hmac("") is None
