"""Fetch MDX content from the Docusaurus repo, chunk + embed it, and persist to Chroma.

Run:  python ingest.py
Re-run any time the source repo's content changes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import chromadb
import requests
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

GITHUB_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def list_mdx_files(content_dir: str) -> list[str]:
    """Return slugs (filenames without .mdx) of content in the given directory."""
    url = GITHUB_API.format(
        repo=config.SOURCE_REPO, branch=config.SOURCE_BRANCH, path=content_dir
    )
    r = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"})
    r.raise_for_status()
    slugs: list[str] = []
    for item in r.json():
        name = item.get("name", "")
        if not name.endswith(".mdx"):
            continue
        if name in config.SKIP_FILES:
            continue
        slugs.append(name.removesuffix(".mdx"))
    return sorted(slugs)


def fetch_file(content_dir: str, slug: str) -> str:
    """Return the raw MDX text for one page, caching to disk so re-runs are fast."""
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
    text = r.text
    cache_path.write_text(text, encoding="utf-8")
    return text


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


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


def build_documents(content_dir: str) -> list[Document]:
    """Fetch every page in the directory and wrap each as a LlamaIndex Document."""
    docs: list[Document] = []
    slugs = list_mdx_files(content_dir)
    print(f"  {content_dir}: {len(slugs)} files")
    for slug in slugs:
        raw = fetch_file(content_dir, slug)
        meta, body = strip_frontmatter(raw)
        url = f"{config.SITE_BASE_URL}/{content_dir}/{slug}"
        docs.append(
            Document(
                text=body,
                metadata={
                    "slug": slug,
                    "content_type": content_dir,
                    "title": meta.get("title", slug),
                    "url": url,
                },
            )
        )
    return docs


def build_index(content_dir: str, client: chromadb.PersistentClient) -> None:
    """Embed every document in `content_dir` and write to a Chroma collection."""
    collection_name = config.COLLECTIONS[content_dir]
    # Drop + recreate for idempotent re-ingests.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = build_documents(content_dir)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
    # Propagate source metadata onto each chunk.
    for node in nodes:
        src = next(
            (d for d in docs if d.text and node.text in d.text), None
        ) or docs[0]
        node.metadata.update(src.metadata)

    VectorStoreIndex(nodes, storage_context=storage_context)
    print(f"  {content_dir}: indexed {len(nodes)} chunks from {len(docs)} pages")


def main() -> int:
    api_key = config.require_api_key()
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=config.GEMINI_EMBED_MODEL, api_key=api_key
    )
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    print(f"Ingesting from {config.SOURCE_REPO}@{config.SOURCE_BRANCH}")
    for content_dir in config.CONTENT_DIRS:
        build_index(content_dir, client)
    print(f"Done. Index persisted to {config.CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
