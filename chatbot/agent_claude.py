"""One chat turn on Claude Haiku: the same tool loop as agent.py, streamed.

It yields the same events and fills the same `TurnResult` as `agent.ChatTurn`, so
the server treats both alike. Cost choices live here and in config.py: the
cheapest Claude (Haiku 5.5), effort "low", a small output cap, no forced tool use
(not allowed on this model), and fewer passages per search.

Thinking blocks come back in `response.content`; that content is replayed
unchanged on the next step, which the API requires.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import anthropic

from chatbot import config
from chatbot.agent import REPROMPT, TurnResult, _SentinelGate
from chatbot.tools import ToolRunner, claude_tool_declarations

RETRYABLE_CODES = {429, 500, 502, 503, 529}

# Appended to the shared system prompt: Haiku likes to announce its searches.
PROMPT_SUFFIX = (
    "\n\nCall tools directly, without announcing it first. Answer in as few words as "
    "the question allows."
)


class ClaudeChatTurn:
    """Streams one answer as (event, data) pairs, like `agent.ChatTurn`."""

    def __init__(
        self,
        client,
        runner: ToolRunner,
        *,
        system_instruction: str,
        model: str = config.CLAUDE_CHAT_MODEL,
        max_steps: int = config.MAX_STEPS,
        max_output_tokens: int = config.CLAUDE_MAX_OUTPUT_TOKENS,
        effort: str = config.CLAUDE_EFFORT,
        retry_backoff: float = 1.5,
    ):
        self._client = client
        self._runner = runner
        self.system_instruction = system_instruction
        self.model = model
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        self.effort = effort
        self.retry_backoff = retry_backoff
        self.result = TurnResult()

    async def stream(self, messages: list[tuple[str, str]]) -> AsyncIterator[tuple[str, dict]]:
        """`messages` are (role, text) pairs, role "user" or "assistant", ending with the user."""
        started = time.perf_counter()
        history: list[dict] = [{"role": role, "content": text} for role, text in messages]
        tools = claude_tool_declarations(self._runner.page_types)
        sent_text = False  # text on screen that a retry would duplicate
        shown = ""  # answer text the client has, kept if the turn fails
        reprompted = False
        try:
            for _ in range(self.max_steps):
                for attempt in (1, 2):
                    gate = _SentinelGate()
                    try:
                        async with self._client.messages.stream(
                            model=self.model,
                            max_tokens=self.max_output_tokens,
                            output_config={"effort": self.effort},
                            system=self.system_instruction,
                            tools=tools,
                            messages=history,
                        ) as stream:
                            async for piece in stream.text_stream:
                                released = gate.feed(piece)
                                if released:
                                    sent_text = True
                                    shown += released
                                    yield "delta", {"text": released}
                            final = await stream.get_final_message()
                        _add_usage(self.result, final)
                        break
                    except anthropic.APIStatusError as exc:
                        if exc.status_code not in RETRYABLE_CODES or sent_text:
                            raise
                        if attempt == 1:
                            await asyncio.sleep(self.retry_backoff)
                            continue
                        self.result.outcome = "busy"
                        self.result.error = f"upstream_busy_{exc.status_code}"
                        return

                calls = [block for block in final.content if block.type == "tool_use"]
                if calls:
                    if gate.released:
                        yield "discard", {}
                        sent_text = False
                        shown = ""
                    history.append({"role": "assistant", "content": final.content})
                    results = []
                    for call in calls:
                        args = dict(call.input or {})
                        self.result.tool_calls.append(call.name)
                        yield "status", {"tool": call.name, "type": args.get("type")}
                        output = await asyncio.to_thread(self._runner.run, call.name, args)
                        results.append(
                            {"type": "tool_result", "tool_use_id": call.id, "content": output}
                        )
                    # All results of one turn go back together in a single user message.
                    history.append({"role": "user", "content": results})
                    continue

                rest = gate.finish()
                if gate.out_of_scope:
                    self.result.outcome = "out_of_scope"
                    return
                if not gate.text.strip():
                    if self.result.tool_calls and not reprompted:
                        # An empty turn can't be replayed; ask once more instead.
                        reprompted = True
                        history.append({"role": "user", "content": REPROMPT})
                        continue
                    self.result.outcome = "error"
                    self.result.error = (
                        "refusal" if final.stop_reason == "refusal" else "empty_response"
                    )
                    return
                if rest:
                    yield "delta", {"text": rest}
                self.result.answer = gate.text.strip()
                self.result.sources = self._runner.cited_sources(self.result.answer)
                self.result.outcome = "answered"
                if self.result.sources:
                    yield "sources", {"items": self.result.sources}
                return

            self.result.outcome = "error"
            self.result.error = "max_steps_exceeded"
        except Exception as exc:
            self.result.outcome = "error"
            self.result.error = f"{type(exc).__name__}: {exc}"
        finally:
            if not self.result.answer and shown.strip():
                self.result.answer = shown.strip()  # aborted or failed part-way
            self.result.top_score = self._runner.top_score
            self.result.latency_ms = int((time.perf_counter() - started) * 1000)


def _add_usage(result: TurnResult, message) -> None:
    """Fold one response's usage into the turn's totals (same columns as Gemini).

    `input_tokens` counts cached input too, as Gemini's does. Claude reports
    output tokens with thinking included, so `thinking_tokens` stays 0.
    """
    result.model_calls += 1
    usage = getattr(message, "usage", None)
    if usage is None:
        return
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    result.input_tokens += (usage.input_tokens or 0) + cached + written
    result.cached_tokens += cached
    result.output_tokens += usage.output_tokens or 0
