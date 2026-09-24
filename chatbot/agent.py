"""One chat turn: the Gemini function-calling loop, streamed.

The model decides when to call the tools; we run them and send the results
back, then stream the final answer. The loop is written out by hand
(automatic function calling off) so every step is visible and testable.

It works with Gemini 2.5 and 3.x: the model's turn is replayed exactly as
returned (Gemini 3 needs its thought signatures back), each tool result
echoes its call id, and temperature is only sent to 2.x models.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from google.genai import errors as genai_errors
from google.genai import types

from chatbot import config
from chatbot.store import Page
from chatbot.tools import ToolRunner, tool_declarations

# The model is told to answer off-topic requests with exactly this string;
# the server swaps it for a translated refusal.
OUT_OF_SCOPE_SENTINEL = "__out_of_scope__"
RETRYABLE_CODES = {429, 503}
HOLD_CHARS = 60  # answer text held back until it clearly isn't the sentinel

_PROMPT = """\
You are Collegesaurus, an assistant that helps Lebanese high-school students \
(and their parents) choose a university or find a scholarship.

What you can look up:
{kinds}

Ground rules:
1. ALWAYS call a tool before answering a factual question. Do not invent \
university names, majors, tuition figures, deadlines or contact details: look \
them up with `search`, or `list_pages` for "what do you cover?" questions.
2. Each search result starts with "page › section [academic year]". When a \
figure comes from an older academic year than the user asked about, say which \
year it is from.
3. When you use a fact from a result, cite its Source URL as a markdown link, \
e.g. [AUB](https://collegesaurus.org/universities/aub). Keep citations short.
4. If the tools return nothing useful, say so plainly. Do not guess.
5. SCOPE — STRICT. You only answer about Lebanese universities, their \
programs, and the scholarships on our pages. For ANY other request (general \
knowledge, non-Lebanese institutions, coding or math help, opinions, jokes, \
translation, weather, news) output EXACTLY this literal string and nothing \
else: __out_of_scope__
   Do not call a tool, apologize or explain. Short conversational glue is \
fine: answer greetings, thanks and acknowledgements with one polite sentence \
that invites a question about Lebanese universities or scholarships, never \
with the sentinel.
   Examples:
     - "Tell me about Harvard" -> __out_of_scope__
     - "Write me a Python function" -> __out_of_scope__
     - "Hi!" -> "Hi! What would you like to know about Lebanese universities \
or scholarships?"
6. Prefer concise, structured answers: short paragraphs, bullet lists for \
options, bold key numbers (deadlines, costs).
7. When asked for a list (majors, programs, scholarships, universities), \
include EVERY item the tools returned, and search again with a more specific \
query if the list looks incomplete.
8. Reply in the language of the user's latest message (English, Arabic or \
French). Keep program and page names as they are written."""


def system_prompt(page_types: dict[str, str], page: Page | None = None) -> str:
    prompt = _PROMPT.format(kinds="\n".join(f"- {k}: {about}" for k, about in page_types.items()))
    if page:
        prompt += (
            f"\n\nThe user is reading this page right now: {page.title} ({page.url}). "
            "If a question doesn't say which university or scholarship it means, "
            "assume it is about this page."
        )
    return prompt


@dataclass
class TurnResult:
    answer: str = ""
    outcome: str = "aborted"  # answered | out_of_scope | busy | error; aborted = client left
    tool_calls: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    top_score: float | None = None
    error: str | None = None
    latency_ms: int = 0


class ChatTurn:
    """Streams one answer as (event, data) pairs; `result` holds the outcome.

    Events: ("status", {tool, type}) while a tool runs, ("delta", {text}) for
    answer text, ("discard", {}) when text already sent turned out to be a
    preamble to a tool call, and ("sources", {items}) at the end.
    """

    def __init__(
        self,
        client,
        runner: ToolRunner,
        *,
        system_instruction: str,
        model: str = config.GEMINI_CHAT_MODEL,
        max_steps: int = config.MAX_STEPS,
        max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
        retry_backoff: float = 1.5,
    ):
        self._client = client
        self._runner = runner
        self.system_instruction = system_instruction
        self.model = model
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        self.retry_backoff = retry_backoff
        self.result = TurnResult()

    async def stream(self, messages: list[tuple[str, str]]) -> AsyncIterator[tuple[str, dict]]:
        """`messages` are (role, text) pairs, role "user" or "assistant", ending with the user."""
        started = time.perf_counter()
        history = [
            types.Content(
                role="user" if role == "user" else "model", parts=[types.Part.from_text(text=text)]
            )
            for role, text in messages
        ]
        request_config = self._config()
        sent_text = False  # text on screen that a retry would duplicate
        try:
            for _ in range(self.max_steps):
                for attempt in (1, 2):
                    parts: list[types.Part] = []
                    calls: list[types.FunctionCall] = []
                    gate = _SentinelGate()
                    try:
                        stream = await self._client.aio.models.generate_content_stream(
                            model=self.model, contents=history, config=request_config
                        )
                        async for chunk in stream:
                            for part in _parts(chunk):
                                parts.append(part)
                                if part.function_call:
                                    calls.append(part.function_call)
                                elif part.text and not part.thought:
                                    released = gate.feed(part.text)
                                    if released:
                                        sent_text = True
                                        yield "delta", {"text": released}
                        break
                    except genai_errors.APIError as exc:
                        if exc.code not in RETRYABLE_CODES or sent_text:
                            raise
                        if attempt == 1:
                            await asyncio.sleep(self.retry_backoff)
                            continue
                        self.result.outcome = "busy"
                        self.result.error = f"upstream_busy_{exc.code}"
                        return

                history.append(types.Content(role="model", parts=parts))
                if calls:
                    if gate.released:
                        yield "discard", {}
                        sent_text = False
                    responses = []
                    for call in calls:
                        args = dict(call.args or {})
                        self.result.tool_calls.append(call.name)
                        yield "status", {"tool": call.name, "type": args.get("type")}
                        output = await asyncio.to_thread(self._runner.run, call.name, args)
                        responses.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=call.id, name=call.name, response={"result": output}
                                )
                            )
                        )
                    history.append(types.Content(role="user", parts=responses))
                    continue

                rest = gate.finish()
                if gate.out_of_scope:
                    self.result.outcome = "out_of_scope"
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
            self.result.top_score = self._runner.top_score
            self.result.latency_ms = int((time.perf_counter() - started) * 1000)

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[types.Tool(function_declarations=tool_declarations(self._runner.page_types))],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=self.max_output_tokens,
            # Gemini 3 deprecates temperature in favour of thinking levels.
            temperature=0.3 if self.model.startswith("gemini-2") else None,
        )


class _SentinelGate:
    """Holds back the start of an answer until it clearly isn't the refusal sentinel."""

    def __init__(self) -> None:
        self.text = ""
        self.released = False  # whether any text has been let through
        self.out_of_scope = False
        self._sent = 0
        self._decided = False

    def feed(self, piece: str) -> str:
        self.text += piece
        if not self._decided and len(self.text.lstrip()) >= HOLD_CHARS:
            self._decide()
        return self._release() if self._decided else ""

    def finish(self) -> str:
        if not self._decided:
            self._decide()
        return self._release()

    def _decide(self) -> None:
        self._decided = True
        self.out_of_scope = OUT_OF_SCOPE_SENTINEL in self.text.lstrip()[:HOLD_CHARS]

    def _release(self) -> str:
        if self.out_of_scope:
            return ""
        out = self.text[self._sent :]
        self._sent = len(self.text)
        self.released = self.released or bool(out)
        return out


def _parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    if not chunk.candidates or not chunk.candidates[0].content:
        return []
    return list(chunk.candidates[0].content.parts or [])
