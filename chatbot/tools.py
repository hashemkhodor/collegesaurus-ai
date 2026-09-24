"""The tools the model can call, and what they retrieved during a turn.

Two generic tools cover every source: `search` (semantic search, optionally
limited to one type of page) and `list_pages`. Their `type` enum comes from the
registered sources, so a new source needs no new tool. Declarations are written
out by hand so the model sees exactly this schema.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
from google.genai import types

from chatbot import config
from chatbot.store import Hit, VectorStore

MAX_SOURCES = 4  # source chips shown under an answer
_URL = re.compile(r"https?://[^\s)\]>\"'<]+")
_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")


def question_locale(text: str) -> str:
    """'ar' for mostly-Arabic questions; everything else searches English pages."""
    return "ar" if len(_ARABIC.findall(text)) > len(_LATIN.findall(text)) else "en"


def tool_declarations(page_types: dict[str, str]) -> list[types.FunctionDeclaration]:
    type_param = types.Schema(
        type=types.Type.STRING,
        enum=list(page_types),
        description="Only this kind of page. "
        + " ".join(f"{kind}: {about}" for kind, about in page_types.items()),
    )
    return [
        types.FunctionDeclaration(
            name="search",
            description=(
                "Search the Collegesaurus knowledge base. Returns the best-matching "
                "passages; each starts with 'page › section [academic year]' followed "
                "by its Source URL. Use specific queries, e.g. 'AUB engineering "
                "tuition' or 'Fulbright eligibility', and search again if the first "
                "results miss part of the question."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="What to look for."),
                    "type": type_param,
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_pages",
            description=(
                "List every page we have, with its URL, optionally only one kind. Use "
                "when the user asks which universities or scholarships we cover."
            ),
            parameters=types.Schema(type=types.Type.OBJECT, properties={"type": type_param}),
        ),
    ]


class ToolRunner:
    """Runs one turn's tool calls against the index and remembers what they found."""

    def __init__(
        self,
        store: VectorStore,
        embed_query: Callable[[str], np.ndarray],
        page_types: dict[str, str],
        *,
        locale: str,
        top_k: int = config.TOP_K,
    ):
        self.store = store
        self.embed_query = embed_query
        self.page_types = page_types
        self.locale = locale
        self.top_k = top_k
        self.hits: list[Hit] = []

    def run(self, name: str, args: dict) -> str:
        if name == "search":
            return self._search(args)
        if name == "list_pages":
            return self._list_pages(args)
        return f"Error: unknown tool {name!r}; use search or list_pages."

    @property
    def top_score(self) -> float | None:
        return max((h.score for h in self.hits), default=None)

    def cited_sources(self, answer: str) -> list[dict[str, str]]:
        """Retrieved pages the answer links to, in the order it mentions them."""
        titles = {h.chunk.url: h.chunk.title for h in self.hits}
        cited: list[dict[str, str]] = []
        for url in _URL.findall(answer):
            url = url.rstrip(".,;:!?")
            if url in titles and all(c["url"] != url for c in cited):
                cited.append({"title": titles[url], "url": url})
        return cited[:MAX_SOURCES]

    def _search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "Error: search needs a non-empty 'query'."
        kinds, error = self._kinds(args)
        if error:
            return error
        hits = self.store.search(
            self.embed_query(query), self.top_k, locale=self.locale, types=kinds
        )
        self.hits.extend(hits)
        if not hits:
            return "No matches."
        blocks = []
        for i, hit in enumerate(hits, 1):
            crumb, _, body = hit.chunk.text.partition("\n\n")
            blocks.append(f"[{i}] {crumb}\nSource: {hit.chunk.url}\n{body}")
        return "\n\n---\n\n".join(blocks)

    def _list_pages(self, args: dict) -> str:
        kinds, error = self._kinds(args)
        if error:
            return error
        pages = self.store.pages(locale=self.locale, types=kinds)
        return "\n".join(f"- {p.title} ({p.url})" for p in pages) or "No pages."

    def _kinds(self, args: dict) -> tuple[set[str] | None, str | None]:
        kind = args.get("type")
        if not kind:
            return None, None
        if kind not in self.page_types:
            return None, f"Error: unknown type {kind!r}; use one of {', '.join(self.page_types)}."
        return {kind}, None
