import pytest
from conftest import busy, call_chunk, local_source, scripted_client, text_chunk

from chatbot.agent import ChatTurn, system_prompt
from chatbot.ingest import Refresher
from chatbot.sources import CollegesaurusCorpus
from chatbot.store import Page
from chatbot.tools import ToolRunner

AUB_URL = "https://collegesaurus.org/universities/aub"
LONG_PREAMBLE = (
    "Let me look that up for you in the Collegesaurus knowledge base, one moment please."
)


@pytest.fixture
def store(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    refresher.refresh_once()
    return refresher.store


def make_turn(client, store, embedder, *, model="gemini-2.5-flash-lite", max_steps=6):
    runner = ToolRunner(store, embedder.embed_query, CollegesaurusCorpus.types, locale="en")
    return ChatTurn(
        client,
        runner,
        model=model,
        system_instruction="You are Collegesaurus.",
        max_steps=max_steps,
        retry_backoff=0,
    )


async def collect(turn, messages=(("user", "What does AUB charge per credit?"),)):
    return [event async for event in turn.stream(list(messages))]


def deltas(events) -> str:
    return "".join(data["text"] for kind, data in events if kind == "delta")


@pytest.mark.anyio
async def test_searches_then_streams_the_answer_with_its_cited_sources(store, embedder):
    client = scripted_client(
        [call_chunk("search", {"query": "AUB tuition per credit", "type": "university"})],
        [
            text_chunk("AUB charges **$1,000 per credit** for engineering programs "),
            text_chunk(f"([AUB]({AUB_URL}))."),
        ],
    )
    turn = make_turn(client, store, embedder)

    events = await collect(turn)

    assert events[0] == ("status", {"tool": "search", "type": "university"})
    answer = f"AUB charges **$1,000 per credit** for engineering programs ([AUB]({AUB_URL}))."
    assert deltas(events) == answer
    sources = [{"title": "AUB — American University of Beirut", "url": AUB_URL}]
    assert events[-1] == ("sources", {"items": sources})
    result = turn.result
    assert (result.outcome, result.answer, result.tool_calls) == ("answered", answer, ["search"])
    assert result.sources == sources
    assert result.top_score > 0


@pytest.mark.anyio
async def test_tool_result_is_sent_back_with_the_call_id_and_the_signed_call(store, embedder):
    client = scripted_client(
        [call_chunk("search", {"query": "AUB tuition per credit"}, call_id="call-7")],
        [text_chunk("AUB charges $1,000 per credit.")],
    )

    await collect(make_turn(client, store, embedder))

    *_, model_turn, tool_turn = client.aio.models.requests[1]["contents"]
    assert model_turn.role == "model"
    assert model_turn.parts[0].function_call.id == "call-7"
    assert model_turn.parts[0].thought_signature == b"signature-1"
    response = tool_turn.parts[0].function_response
    assert (response.id, response.name) == ("call-7", "search")
    assert f"Source: {AUB_URL}" in response.response["result"]


@pytest.mark.anyio
async def test_long_preamble_before_a_tool_call_is_discarded(store, embedder):
    client = scripted_client(
        [text_chunk(LONG_PREAMBLE), call_chunk("search", {"query": "AUB tuition"})],
        [text_chunk("AUB charges $1,000 per credit.")],
    )
    turn = make_turn(client, store, embedder)

    events = await collect(turn)

    kinds = [kind for kind, _ in events]
    assert kinds.index("discard") < kinds.index("status")
    assert events[0] == ("delta", {"text": LONG_PREAMBLE})
    assert turn.result.answer == "AUB charges $1,000 per credit."


@pytest.mark.anyio
async def test_short_preamble_never_reaches_the_client(store, embedder):
    client = scripted_client(
        [text_chunk("Let me check."), call_chunk("search", {"query": "AUB tuition"})],
        [text_chunk("AUB charges $1,000 per credit.")],
    )

    events = await collect(make_turn(client, store, embedder))

    assert "discard" not in [kind for kind, _ in events]
    assert deltas(events) == "AUB charges $1,000 per credit."


@pytest.mark.anyio
async def test_out_of_scope_sentinel_is_never_streamed(store, embedder):
    client = scripted_client([text_chunk("  `__out_of"), text_chunk("_scope__`")])
    turn = make_turn(client, store, embedder)

    events = await collect(turn, [("user", "Write me a poem about Paris")])

    assert deltas(events) == ""
    assert turn.result.outcome == "out_of_scope"


@pytest.mark.anyio
async def test_short_answer_is_released_when_the_stream_ends(store, embedder):
    client = scripted_client([text_chunk("Hi! What would you like to know?")])
    turn = make_turn(client, store, embedder)

    events = await collect(turn, [("user", "Hi")])

    assert events == [("delta", {"text": "Hi! What would you like to know?"})]
    assert (turn.result.outcome, turn.result.tool_calls) == ("answered", [])


@pytest.mark.anyio
async def test_busy_model_is_retried_once_before_any_text_was_sent(store, embedder):
    client = scripted_client(busy(503), [text_chunk("Hello, ask me about universities.")])
    turn = make_turn(client, store, embedder)

    events = await collect(turn, [("user", "Hi")])

    assert deltas(events) == "Hello, ask me about universities."
    assert turn.result.outcome == "answered"
    assert len(client.aio.models.requests) == 2


@pytest.mark.anyio
async def test_model_still_busy_after_the_retry_reports_busy(store, embedder):
    client = scripted_client(busy(503), busy(429))
    turn = make_turn(client, store, embedder)

    events = await collect(turn)

    assert events == []
    assert (turn.result.outcome, turn.result.error) == ("busy", "upstream_busy_429")


@pytest.mark.anyio
async def test_error_after_text_was_sent_is_not_retried(store, embedder):
    long_text = "AUB charges $1,000 per credit for its engineering programs, and more. "
    client = scripted_client([text_chunk(long_text), busy(503)])
    turn = make_turn(client, store, embedder)

    events = await collect(turn)

    assert deltas(events) == long_text
    assert turn.result.outcome == "error"
    assert len(client.aio.models.requests) == 1


@pytest.mark.anyio
async def test_gives_up_after_max_steps_of_tool_calls(store, embedder):
    client = scripted_client(*[[call_chunk("search", {"query": f"q{i}"})] for i in range(3)])
    turn = make_turn(client, store, embedder, max_steps=3)

    await collect(turn)

    assert (turn.result.outcome, turn.result.error) == ("error", "max_steps_exceeded")
    assert turn.result.tool_calls == ["search", "search", "search"]


@pytest.mark.anyio
async def test_history_is_sent_as_alternating_user_and_model_turns(store, embedder):
    client = scripted_client([text_chunk("AUB charges $1,000 per credit.")])
    messages = [("user", "Hi"), ("assistant", "Hello! Ask me anything."), ("user", "AUB fees?")]

    await collect(make_turn(client, store, embedder), messages)

    contents = client.aio.models.requests[0]["contents"]
    assert [(c.role, c.parts[0].text) for c in contents] == [
        ("user", "Hi"),
        ("model", "Hello! Ask me anything."),
        ("user", "AUB fees?"),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "temperature"), [("gemini-2.5-flash-lite", 0.3), ("gemini-3.5-flash-lite", None)]
)
async def test_temperature_is_only_sent_to_models_that_accept_it(
    store, embedder, model, temperature
):
    client = scripted_client([text_chunk("Hello.")])

    await collect(make_turn(client, store, embedder, model=model), [("user", "Hi")])

    request = client.aio.models.requests[0]
    assert request["model"] == model
    assert request["config"].temperature == temperature
    assert {d.name for d in request["config"].tools[0].function_declarations} == {
        "search",
        "list_pages",
    }


def test_system_prompt_names_the_page_the_user_is_viewing():
    page = Page("university/aub", "university", "AUB — American University of Beirut", AUB_URL)

    with_page = system_prompt(CollegesaurusCorpus.types, page)
    without_page = system_prompt(CollegesaurusCorpus.types)

    assert f"reading this page right now: AUB — American University of Beirut ({AUB_URL})" in (
        with_page
    )
    assert "reading this page" not in without_page
    assert "__out_of_scope__" in without_page


@pytest.mark.anyio
async def test_an_empty_model_reply_is_an_error_not_an_answer(store, embedder):
    client = scripted_client(
        [call_chunk("search", {"query": "AUB tuition"})],
        [],  # e.g. blocked for safety, or the known empty reply after a tool result
    )
    turn = make_turn(client, store, embedder)

    events = await collect(turn)

    assert deltas(events) == ""
    assert (turn.result.outcome, turn.result.error) == ("error", "empty_response")


@pytest.mark.anyio
async def test_text_already_shown_is_kept_when_the_turn_fails(store, embedder):
    long_text = "AUB charges $1,000 per credit for its engineering programs, and more. "
    client = scripted_client([text_chunk(long_text), busy(503)])
    turn = make_turn(client, store, embedder)

    await collect(turn)

    assert turn.result.outcome == "error"
    assert turn.result.answer == long_text.strip()
