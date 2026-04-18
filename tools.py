"""Agent tools. Each tool is a small, named function the ReAct loop can call.

Design notes
------------
- Keep the surface narrow: the agent should have a couple of focused tools, not
  one god-tool. This makes the reasoning trace readable and lets us measure
  which tools actually get used.
- Every tool returns a plain string. LlamaIndex will feed that string back into
  the model's next turn.
- Retrieval tools always include a source URL so the agent can cite it.
"""

from __future__ import annotations

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def _load_retriever(collection_name: str, top_k: int = config.TOP_K):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_retriever(similarity_top_k=top_k)


def _format_hits(hits) -> str:
    if not hits:
        return "No matches."
    lines = []
    for i, node in enumerate(hits, 1):
        meta = node.metadata or {}
        title = meta.get("title", "(untitled)")
        url = meta.get("url", "")
        snippet = node.get_content().strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        lines.append(f"[{i}] {title}\n  Source: {url}\n  Excerpt: {snippet}")
    return "\n\n".join(lines)


def search_universities(query: str) -> str:
    """Search the Lebanese-universities knowledge base.

    Returns top matches with title, source URL, and an excerpt. Use for
    questions about specific schools (AUB, LAU, USJ, etc.), their majors,
    tuition, admissions requirements, or contacts.
    """
    retriever = _load_retriever(config.COLLECTIONS["universities"])
    return _format_hits(retriever.retrieve(query))


def search_scholarships(query: str) -> str:
    """Search the external-scholarships knowledge base.

    Returns top matches with title, source URL, and an excerpt. Use for
    questions about specific scholarships (Fulbright, USAID USP, LIFE, etc.),
    eligibility, deadlines, benefits, or how to apply.
    """
    retriever = _load_retriever(config.COLLECTIONS["scholarships"])
    return _format_hits(retriever.retrieve(query))


def _list_collection(collection_name: str, content_dir: str) -> str:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection(collection_name)
    # Chroma has no .distinct() — pull metadata in one batch and dedupe by slug.
    result = collection.get(include=["metadatas"])
    seen: dict[str, str] = {}
    for meta in result.get("metadatas") or []:
        slug = meta.get("slug")
        if slug and slug not in seen:
            seen[slug] = meta.get("title", slug)
    if not seen:
        return f"No {content_dir} indexed yet."
    lines = [f"- {title} ({config.SITE_BASE_URL}/{content_dir}/{slug})"
             for slug, title in sorted(seen.items())]
    return "\n".join(lines)


def list_universities() -> str:
    """List every university we have a page for, with its site URL.

    Use when the user asks "what schools do you have?" or "list all unis".
    """
    return _list_collection(config.COLLECTIONS["universities"], "universities")


def list_scholarships() -> str:
    """List every scholarship we have a page for, with its site URL.

    Use when the user asks "what scholarships are available?".
    """
    return _list_collection(config.COLLECTIONS["scholarships"], "scholarships")


def build_tools() -> list[FunctionTool]:
    """Wrap each tool for LlamaIndex. Docstrings become tool descriptions."""
    return [
        FunctionTool.from_defaults(fn=search_universities),
        FunctionTool.from_defaults(fn=search_scholarships),
        FunctionTool.from_defaults(fn=list_universities),
        FunctionTool.from_defaults(fn=list_scholarships),
    ]
