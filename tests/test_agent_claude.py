import pytest
from conftest import claude_busy, claude_call, claude_text, local_source, scripted_claude

from chatbot.agent_claude import ClaudeChatTurn
from chatbot.ingest import Refresher
from chatbot.sources import CollegesaurusCorpus
from chatbot.tools import ToolRunner

AUB_URL = "https://collegesaurus.org/universities/aub"
LONG_PREAMBLE = "Let me look that up for you in the Collegesaurus knowledge base, one moment."


@pytest.fixture
def store(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    refresher.refresh_once()
    return refresher.store


def make_turn(client, store, embedder, *, max_steps=6):
    runner = ToolRunner(store, embedder.embed_query, CollegesaurusCorpus.types, locale="en")
    return ClaudeChatTurn(
        client,
        runner,
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
    client = scripted_claude(
        [claude_call("search", {"query": "AUB tuition per credit", "type": "university"})],
        [claude_text(f"AUB charges **$1,000 per credit** ([AUB]({AUB_URL})), as listed.")],
    )
    turn = make_turn(client, store, embedder)

    events = await collect(turn)

    assert events[0] == ("status", {"tool": "search", "type": "university"})
    answer = f"AUB charges **$1,000 per credit** ([AUB]({AUB_URL})), as listed."
    assert deltas(events) == answer
    sources = [{"title": "AUB — American University of Beirut", "url": AUB_URL}]
    assert events[-1] == ("sources", {"items": sources})
    result = turn.result
    assert (result.outcome, result.answer, result.tool_calls) == ("answered", answer, ["search"])
    assert result.sources == sources
    assert result.top_score > 0


@pytest.mark.anyio
async def test_token_usage_is_summed_over_the_turn(store, embedder):
    client = scripted_claude(
        [claude_call("search", {"query": "AUB tuition"})],
        [claude_text("AUB charges $1,000 per credit, according to its page.")],
    )
    turn = make_turn(client, store, embedder)

    await collect(turn)

    result = turn.result
    assert result.model_calls == 2
    assert result.input_tokens == 2 * (100 + 20)  # cached input counts, as with Gemini
    assert (result.cached_tokens, result.output_tokens, result.thinking_tokens) == (40, 60, 0)


@pytest.mark.anyio
async def test_request_is_cheap_and_declares_the_tools(store, embedder):
    client = scripted_claude([claude_text("Hello! What would you like to know?" * 3)])

    await collect(make_turn(client, store, embedder), [("user", "Hi")])

    request = client.messages.requests[0]
    assert request["model"] == "claude-haiku-5-5"
    assert request["output_config"] == {"effort": "low"}
    assert request["max_tokens"] <= 2048
    assert request["system"] == "You are Collegesaurus."
    names = {tool["name"]: tool for tool in request["tools"]}
    assert set(names) == {"search", "list_pages"}
    search = names["search"]["input_schema"]
    assert search["required"] == ["query"]
    assert search["properties"]["type"]["enum"] == list(CollegesaurusCorpus.types)
    assert "tool_choice" not in request  # forced tool use is not allowed on this model
    assert "temperature" not in request


@pytest.mark.anyio
async def test_tool_result_goes_back_with_the_call_id_and_the_model_turn(store, embedder):
    first = [claude_call("search", {"query": "AUB tuition per credit"}, call_id="toolu-7")]
    client = scripted_claude(first, [claude_text("AUB charges $1,000 per credit, per its page.")])

    await collect(make_turn(client, store, embedder))

    *_, model_turn, tool_turn = client.messages.requests[1]["messages"]
    assert model_turn == {"role": "assistant", "content": first}
    result = tool_turn["content"][0]
    assert (tool_turn["role"], result["type"], result["tool_use_id"]) == (
        "user",
        "tool_result",
        "toolu-7",
    )
    assert f"Source: {AUB_URL}" in result["content"]


@pytest.mark.anyio
async def test_parallel_tool_calls_return_all_results_in_one_message(store, embedder):
    client = scripted_claude(
        [
            claude_call("search", {"query": "AUB tuition"}, call_id="a"),
            claude_call("list_pages", {}, call_id="b"),
        ],
        [claude_text("Here is what we have on AUB and the other pages we cover.")],
    )
    turn = make_turn(client, store, embedder)

    await collect(turn)

    *_, tool_turn = client.messages.requests[1]["messages"]
    assert [r["tool_use_id"] for r in tool_turn["content"]] == ["a", "b"]
    assert turn.result.tool_calls == ["search", "list_pages"]


@pytest.mark.anyio
async def test_long_preamble_before_a_tool_call_is_discarded(store, embedder):
    client = scripted_claude(
        [claude_text(LONG_PREAMBLE), claude_call("search", {"query": "AUB tuition"})],
        [claude_text("AUB charges $1,000 per credit, according to its page.")],
    )
    turn = make_turn(client, store, embedder)

    events = await collect(turn)

    kinds = [kind for kind, _ in events]
    assert kinds.index("discard") < kinds.index("status")
    assert turn.result.answer == "AUB charges $1,000 per credit, according to its page."


@pytest.mark.anyio
async def test_out_of_scope_sentinel_is_never_streamed(store, embedder):
    client = scripted_claude([claude_text("__out_of_scope__")])
    turn = make_turn(client, store, embedder)

    events = await collect(turn, [("user", "Tell me about Harvard")])

    assert deltas(events) == ""
    assert turn.result.outcome == "out_of_scope"


@pytest.mark.anyio
async def test_empty_reply_after_a_tool_result_is_reprompted_once(store, embedder):
    client = scripted_claude(
        [claude_call("search", {"query": "AUB"})],
        [],
        [claude_text("AUB is the American University of Beirut, a private university.")],
    )
    turn = make_turn(client, store, embedder)

    await collect(turn)

    assert turn.result.outcome == "answered"
    last = client.messages.requests[2]["messages"][-1]
    assert last["role"] == "user" and "Answer my question" in str(last["content"])


@pytest.mark.anyio
async def test_overloaded_twice_is_busy(store, embedder):
    client = scripted_claude(claude_busy(529), claude_busy(529))
    turn = make_turn(client, store, embedder)

    events = await collect(turn)

    assert events == []
    assert (turn.result.outcome, turn.result.error) == ("busy", "upstream_busy_529")


@pytest.mark.anyio
async def test_overloaded_once_is_retried(store, embedder):
    client = scripted_claude(
        claude_busy(529), [claude_text("Hello! What would you like to know about universities?")]
    )
    turn = make_turn(client, store, embedder)

    await collect(turn, [("user", "Hi")])

    assert turn.result.outcome == "answered"


@pytest.mark.anyio
async def test_gives_up_after_max_steps(store, embedder):
    client = scripted_claude(*[[claude_call("search", {"query": "AUB"})]] * 3)
    turn = make_turn(client, store, embedder, max_steps=3)

    await collect(turn)

    assert (turn.result.outcome, turn.result.error) == ("error", "max_steps_exceeded")
