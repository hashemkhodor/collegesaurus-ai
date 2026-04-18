"""Agent tools — explicit JSON-schema declarations + executor registry.

Why not rely on the SDK's automatic function calling?
-----------------------------------------------------
The `google-genai` auto-declaration path introspects Python signatures and
docstrings to build `FunctionDeclaration`s. Silent failures are possible
(e.g., `from __future__ import annotations` turning type hints into strings),
and when declaration fails the tool is simply dropped — the model then
reports "no tool available" and we have no hook to fix it. Declaring schemas
by hand is more verbose but leaves nothing to chance: what you write here is
exactly what the model sees.
"""

import chromadb
from google import genai
from google.genai import types

import config


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

# Module-level Gemini client so its underlying httpx pool stays alive.
# Constructing the Client inside a function call occasionally left the pool
# torn down before the first request, raising "Cannot send a request, as the
# client has been closed". Eager construction sidesteps that entirely.
_GEMINI = genai.Client(api_key=config.require_api_key())

_chroma_client = None


def _chroma():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma_client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed_query(text):
    result = _GEMINI.models.embed_content(
        model=config.GEMINI_EMBED_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def _search(collection_name, query, top_k):
    collection = _chroma().get_collection(collection_name)
    qvec = _embed_query(query)
    res = collection.query(query_embeddings=[qvec], n_results=top_k)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    if not docs:
        return "No matches."
    # Return the FULL chunk text — not a short excerpt. A 500-char cap was
    # truncating long tables (e.g. MSFEA majors list) so the model only saw
    # the first few rows. Gemini 2.5 Flash-Lite has a 1M-token window;
    # 15 chunks x ~1 KB each is trivially cheap.
    lines = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        lines.append(
            "[{}] {}\nSource: {}\n{}".format(
                i,
                meta.get("title", "(untitled)"),
                meta.get("url", ""),
                doc.strip(),
            )
        )
    return "\n\n---\n\n".join(lines)


def _list(collection_name, content_dir):
    collection = _chroma().get_collection(collection_name)
    result = collection.get(include=["metadatas"])
    seen = {}
    for meta in result.get("metadatas") or []:
        slug = meta.get("slug")
        if slug and slug not in seen:
            seen[slug] = meta.get("title", slug)
    if not seen:
        return "No {} indexed yet.".format(content_dir)
    return "\n".join(
        "- {} ({}/{})".format(title, config.SITE_BASE_URL + "/" + content_dir, slug)
        for slug, title in sorted(seen.items())
    )


# ---------------------------------------------------------------------------
# Tool functions (callable by the executor)
# ---------------------------------------------------------------------------

def search_universities(query):
    return _search(config.COLLECTIONS["universities"], query, config.TOP_K)


def search_scholarships(query):
    return _search(config.COLLECTIONS["scholarships"], query, config.TOP_K)


def list_universities():
    return _list(config.COLLECTIONS["universities"], "universities")


def list_scholarships():
    return _list(config.COLLECTIONS["scholarships"], "scholarships")


# ---------------------------------------------------------------------------
# Explicit FunctionDeclarations. The model sees exactly this.
# ---------------------------------------------------------------------------

_QUERY_PARAM = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "query": types.Schema(
            type=types.Type.STRING,
            description="Natural-language search query.",
        ),
    },
    required=["query"],
)

_NO_PARAMS = types.Schema(type=types.Type.OBJECT, properties={})

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_universities",
        description=(
            "Search the Lebanese-universities knowledge base. Use for "
            "questions about specific schools (AUB, LAU, USJ, etc.), their "
            "majors, tuition, admissions requirements, or contacts. Returns "
            "up to 5 ranked matches, each with a title, source URL, and "
            "excerpt."
        ),
        parameters=_QUERY_PARAM,
    ),
    types.FunctionDeclaration(
        name="search_scholarships",
        description=(
            "Search the external-scholarships knowledge base. Use for "
            "questions about specific scholarships (Fulbright, USAID USP, "
            "LIFE, etc.), eligibility, deadlines, benefits, or how to apply. "
            "Returns up to 5 ranked matches, each with a title, source URL, "
            "and excerpt."
        ),
        parameters=_QUERY_PARAM,
    ),
    types.FunctionDeclaration(
        name="list_universities",
        description=(
            "List every Lebanese university we have a page for, with its "
            "site URL. Use when the user wants the catalog or asks what "
            "schools are available."
        ),
        parameters=_NO_PARAMS,
    ),
    types.FunctionDeclaration(
        name="list_scholarships",
        description=(
            "List every scholarship we have a page for, with its site URL. "
            "Use when the user asks what scholarships are available."
        ),
        parameters=_NO_PARAMS,
    ),
]


# Name → callable dispatch table used by the agent's tool-call loop.
TOOL_REGISTRY = {
    "search_universities": search_universities,
    "search_scholarships": search_scholarships,
    "list_universities": list_universities,
    "list_scholarships": list_scholarships,
}
