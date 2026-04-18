"""Fetch MDX content from the Docusaurus repo, chunk, embed, persist to Chroma.

Run:  python ingest.py

Dependencies:
- chromadb for the vector store
- google-genai for embeddings (gemini-embedding-001)
- requests for raw GitHub fetches

No LlamaIndex — ~50 lines of actual logic.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import chromadb
import requests
from google import genai
from google.genai import types

import config

GITHUB_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def list_mdx_files(content_dir: str) -> list[str]:
    """Return slugs (filenames without .mdx) for every file in the directory."""
    url = GITHUB_API.format(
        repo=config.SOURCE_REPO, branch=config.SOURCE_BRANCH, path=content_dir
    )
    r = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"})
    r.raise_for_status()
    slugs: list[str] = []
    for item in r.json():
        name = item.get("name", "")
        if not name.endswith(".mdx") or name in config.SKIP_FILES:
            continue
        slugs.append(name.removesuffix(".mdx"))
    return sorted(slugs)


def fetch_file(content_dir: str, slug: str) -> str:
    """Return raw MDX text, caching to disk so re-runs are fast."""
    cache_dir = config.SOURCE_CACHE_DIR / content_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{slug}.mdx"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = RAW_URL.format(
        repo=config.SOURCE_REPO,
        branch=config.SOURCE_BRANCH,
        path=f"{content_dir}/{slug}.mdx",
    )
    r = requests.get(url)
    r.raise_for_status()
    cache_path.write_text(r.text, encoding="utf-8")
    return r.text


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
HEADING_RE = re.compile(r"(?=^##\s)", re.MULTILINE)


def strip_frontmatter(text: str) -> tuple[dict, str]:
    """Return ({title, sidebar_label, ...}, body_without_frontmatter)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, text[m.end():]


def split_markdown(text: str, target: int, overlap: int) -> list[str]:
    """Split a markdown document into roughly `target`-sized chunks.

    Strategy: split by top-level `## heading` sections first, then fall back
    to paragraph packing for sections that are still too long. Adds a small
    overlap by prefixing each chunk with the tail of the previous one so
    retrieval hits straddling section boundaries survive.
    """
    sections = [s for s in HEADING_RE.split(text) if s.strip()]
    chunks: list[str] = []
    for section in sections:
        if len(section) <= target:
            chunks.append(section.strip())
            continue
        # Pack paragraphs until we hit the target.
        paras = section.split("\n\n")
        buf = ""
        for p in paras:
            if len(buf) + len(p) + 2 <= target or not buf:
                buf = f"{buf}\n\n{p}" if buf else p
            else:
                chunks.append(buf.strip())
                buf = p
        if buf:
            chunks.append(buf.strip())

    if overlap <= 0 or len(chunks) < 2:
        return chunks

    # Prepend the last `overlap` characters of the previous chunk.
    overlapped: list[str] = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        prefix = prev[-overlap:]
        overlapped.append(f"{prefix}\n\n{curr}".strip())
    return overlapped


def build_chunks(content_dir: str) -> list[dict]:
    """Fetch every page and return a flat list of {text, metadata} chunk dicts."""
    out: list[dict] = []
    slugs = list_mdx_files(content_dir)
    print(f"  {content_dir}: {len(slugs)} files")
    for slug in slugs:
        raw = fetch_file(content_dir, slug)
        meta, body = strip_frontmatter(raw)
        base_meta = {
            "slug": slug,
            "content_type": content_dir,
            "title": meta.get("title", slug),
            "url": f"{config.SITE_BASE_URL}/{content_dir}/{slug}",
        }
        for chunk_text in split_markdown(
            body, target=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP
        ):
            out.append({"text": chunk_text, "metadata": dict(base_meta)})
    return out


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def embed_batch(
    gclient: genai.Client, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """Call Gemini embed_content on up to EMBED_BATCH_SIZE texts at once."""
    result = gclient.models.embed_content(
        model=config.GEMINI_EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]


def embed_all(gclient: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a whole list by batching into chunks of EMBED_BATCH_SIZE."""
    vectors: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, config.EMBED_BATCH_SIZE):
        batch = texts[i : i + config.EMBED_BATCH_SIZE]
        vectors.extend(embed_batch(gclient, batch))
        print(f"    embedded {min(i + len(batch), total)}/{total}")
        # Light throttle between batches so we stay well under per-minute caps.
        if i + config.EMBED_BATCH_SIZE < total:
            time.sleep(0.5)
    return vectors


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def build_collection(
    gclient: genai.Client,
    chroma_client: chromadb.PersistentClient,
    content_dir: str,
) -> None:
    """Fetch → chunk → embed → write one Chroma collection, idempotently."""
    collection_name = config.COLLECTIONS[content_dir]
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(collection_name)

    chunks = build_chunks(content_dir)
    if not chunks:
        print(f"  {content_dir}: nothing to index")
        return

    embeddings = embed_all(gclient, [c["text"] for c in chunks])
    collection.add(
        ids=[f"{content_dir}-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )
    print(f"  {content_dir}: indexed {len(chunks)} chunks")


def main() -> int:
    api_key = config.require_api_key()
    gclient = genai.Client(api_key=api_key)

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    print(f"Ingesting from {config.SOURCE_REPO}@{config.SOURCE_BRANCH}")
    for content_dir in config.CONTENT_DIRS:
        build_collection(gclient, chroma_client, content_dir)
    print(f"Done. Index persisted to {config.CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
