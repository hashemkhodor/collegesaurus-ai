"""Shared test doubles. Only external services (Gemini, Supabase, HTTP) are faked."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from chatbot.sources import CollegesaurusCorpus

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministic bag-of-words vectors standing in for the Gemini embedding API.

    Texts sharing words get similar vectors, so search results are meaningful.
    Every document text it embeds is recorded in `embedded`.
    """

    model = "fake-embedding"
    dim = 64

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else np.eye(self.dim, dtype=np.float32)[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.embedded.extend(texts)
        return np.stack([self.vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self.vector(text)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Writable copies of the fixture corpora, so a test can edit a page."""
    for name in ("corpus.en.json", "corpus.ar.json"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    return tmp_path


def local_source(directory: Path) -> CollegesaurusCorpus:
    return CollegesaurusCorpus(
        {"en": str(directory / "corpus.en.json"), "ar": str(directory / "corpus.ar.json")}
    )


# --- Gemini streaming API stand-ins (external service) -----------------------


def text_chunk(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ]
    )


def call_chunk(name: str, args: dict, call_id: str = "call-1") -> types.GenerateContentResponse:
    part = types.Part(
        function_call=types.FunctionCall(name=name, args=args, id=call_id),
        thought_signature=b"signature-1",
    )
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))]
    )


def usage_chunk(
    prompt: int, output: int, *, thoughts: int | None = None, cached: int | None = None
) -> types.GenerateContentResponse:
    """The usage report Gemini sends with the last chunk of a streamed call."""
    return types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt,
            candidates_token_count=output,
            thoughts_token_count=thoughts,
            cached_content_token_count=cached,
        )
    )


def busy(code: int = 503) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": "overloaded"}})


class ScriptedModels:
    """Stands in for client.aio.models: each streaming call plays the next script.

    A script is a list of response chunks (an exception in the list is raised
    mid-stream), or an exception raised when the call is made.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests = []

    async def generate_content_stream(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        script = self.scripts.pop(0)
        if isinstance(script, Exception):
            raise script

        async def chunks():
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item

        return chunks()


def scripted_client(*scripts):
    return SimpleNamespace(aio=SimpleNamespace(models=ScriptedModels(scripts)))


# --- Anthropic streaming API stand-ins (external service) --------------------


def claude_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def claude_call(name: str, args: dict, call_id: str = "toolu-1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)


def claude_busy(status: int = 529) -> anthropic.APIStatusError:
    import httpx2

    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.test"))
    return anthropic.APIStatusError("overloaded", response=response, body=None)


class FakeClaudeStream:
    def __init__(self, blocks, stop_reason):
        self._blocks = blocks
        self._stop_reason = stop_reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def texts():
            for block in self._blocks:
                if isinstance(block, Exception):
                    raise block
                if block.type == "text":
                    yield block.text

        return texts()

    async def get_final_message(self):
        blocks = [b for b in self._blocks if not isinstance(b, Exception)]
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else self._stop_reason
        return SimpleNamespace(content=blocks, stop_reason=stop)


class ScriptedClaude:
    """Stands in for AsyncAnthropic.messages: each .stream() call plays the next script.

    A script is a list of content blocks (an exception in the list is raised
    mid-stream), or an exception raised when the stream is opened.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests = []

    def stream(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        script = self.scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        return FakeClaudeStream(script, "end_turn")


def scripted_claude(*scripts):
    return SimpleNamespace(messages=ScriptedClaude(scripts))
