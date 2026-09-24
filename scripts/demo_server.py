"""Run the chatbot locally without a Gemini key, to try the UI end to end.

    .venv/bin/python -m scripts.demo_server [--site http://localhost:3000] [--port 8000]

Everything is the real app: the corpus source (reading --site, e.g. a local
`npm run serve` of the collegesaurus site), chunking, search, the server and
the chat page. Only the Gemini API is replaced: a word-hash embedder, and a
scripted "model" that searches for the question and streams a short summary
of the top results with their source links. So answers are canned; greetings
and off-topic questions (weather, code, poems…) show the real behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from types import SimpleNamespace

import numpy as np
import uvicorn
from google.genai import types

from chatbot.ingest import Refresher
from chatbot.logging_store import ChatLogger
from chatbot.server import Services, create_app
from chatbot.sources import CollegesaurusCorpus

OFF_TOPIC = re.compile(r"\b(weather|python|code|poem|joke|harvard)\b|قصيدة|طقس|نكتة", re.I)
GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|bonjour|merci)\b|^\s*(مرحبا|أهلا|شكرا)", re.I)
COVERAGE = re.compile(r"\b(cover|list)\b|تغطي|couvrez", re.I)
SCHOLARSHIP = re.compile(r"scholarship|bourse|منح", re.I)
DEMO_NOTE = "*Demo mode: a canned summary of the top search results, not a real model answer.*"
RESULT_NUMBER = re.compile(r"^\[\d+\]\s*")


# Words too common to help a word-overlap search pick the right page.
STOPWORDS = set(
    "the and for are what does how which who can with from about that this you your our "
    "much many any there their its per les des est une que qui pour dans sur avec "
    "ما هي هو في من على عن إلى التي الذي كم هل".split()
)


class WordEmbedder:
    """TF-IDF bag-of-words vectors hashed into 8192 dimensions; needs no API.

    Document frequencies are learned from the texts it embeds (the whole corpus
    on the first build), so rare words like "LAU" outweigh common ones.
    """

    model = "demo-word-hash"
    dim = 8192

    def __init__(self) -> None:
        self._df = np.zeros(self.dim, dtype=np.float32)
        self._docs = 0

    def _buckets(self, text: str) -> list[int]:
        words = re.findall(r"\w+", text.lower())
        return [
            int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
            for w in words
            if (len(w) > 2 or w.isdigit()) and w not in STOPWORDS
        ]

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for bucket in self._buckets(text):
            v[bucket] += 1.0
        v *= np.log((1 + self._docs) / (1 + self._df)) + 1
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        for text in texts:
            self._df[list(set(self._buckets(text)))] += 1
        self._docs += len(texts)
        return np.array([self._vector(t) for t in texts], dtype=np.float32).reshape(-1, self.dim)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class DemoModels:
    """Stands in for client.aio.models.generate_content_stream."""

    async def generate_content_stream(self, *, model, contents, config):
        last = contents[-1]
        results = [p.function_response for p in last.parts or [] if p.function_response]
        if results:
            return _text_stream(_summary(results[0].name, results[0].response["result"]))
        question = last.parts[0].text or ""
        if OFF_TOPIC.search(question):
            return _text_stream("__out_of_scope__")
        if GREETING.search(question):
            return _text_stream(
                "Hi! This is demo mode: ask about a Lebanese university, a major or a "
                "scholarship and I'll show what the search finds."
            )
        kind = "scholarship" if SCHOLARSHIP.search(question) else "university"
        if COVERAGE.search(question):
            call = types.FunctionCall(name="list_pages", args={"type": kind}, id="demo-list")
        else:
            args = {"query": question} | ({"type": kind} if kind == "scholarship" else {})
            call = types.FunctionCall(name="search", args=args, id="demo-search")
        return _parts_stream([types.Part(function_call=call)])


def _summary(tool: str, result: str) -> str:
    if tool == "list_pages":
        return f"*Demo mode: every page I have.*\n\n{result}"
    if result.startswith(("No matches", "Error")):
        return "*Demo mode:* the search found nothing for that question."
    parts = [DEMO_NOTE]
    for block in result.split("\n\n---\n\n")[:3]:
        crumb, _, rest = block.partition("\n")
        source, _, body = rest.partition("\n")
        url = source.removeprefix("Source: ").strip()
        title = RESULT_NUMBER.sub("", crumb)
        parts.append(f"**{title}**\n\n{_excerpt(body)}\n\n[Source]({url})")
    return "\n\n".join(parts)


def _excerpt(body: str, limit: int = 450) -> str:
    lines: list[str] = []
    size = 0
    for line in body.splitlines():
        if lines and size + len(line) > limit:
            break
        lines.append(line)
        size += len(line)
    return "\n".join(lines)


async def _text_stream(text: str):
    step = max(1, len(text) // 6)
    for start in range(0, len(text), step):
        await asyncio.sleep(0.05)  # so the streaming is visible
        yield _response([types.Part(text=text[start : start + step])])


async def _parts_stream(parts: list[types.Part]):
    yield _response(parts)


def _response(parts: list[types.Part]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m scripts.demo_server")
    parser.add_argument("--site", default="http://localhost:3000", help="site serving /chatbot/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    site = args.site.rstrip("/")
    source = CollegesaurusCorpus(
        {"en": f"{site}/chatbot/corpus.json", "ar": f"{site}/ar/chatbot/corpus.json"},
        version=f"{site}/chatbot/version.json",
    )
    embedder = WordEmbedder()
    refresher = Refresher([source], embedder, store=None)
    if refresher.refresh_once():
        print(f"Indexed {refresher.store.meta['chunks']} chunks from {site}", flush=True)
    else:
        print(f"No index yet ({refresher.last_error}); retrying every minute", flush=True)

    services = Services(
        refresher=refresher,
        embedder=embedder,
        genai_client=SimpleNamespace(aio=SimpleNamespace(models=DemoModels())),
        logger=ChatLogger("", ""),  # logging off
        page_types=source.types,
        model="demo",
    )
    uvicorn.run(create_app(services), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
