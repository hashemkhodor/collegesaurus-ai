"""Build the search index from the sources, and keep it fresh.

    python -m chatbot.ingest --out data/index.db          # from the live site
    python -m chatbot.ingest --corpus en=corpus.en.json --corpus ar=corpus.ar.json
    python -m chatbot.ingest --query "AUB tuition"        # also print the best matches

The deploy workflow runs the first form to bake a snapshot into the image.
The running server calls Refresher.refresh_once() every config.POLL_SECONDS:
a source whose fingerprint is unchanged costs one tiny request, and a changed
source re-embeds only the chunks whose text changed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from google import genai
from google.genai import types

from chatbot import config
from chatbot.chunking import chunk_document
from chatbot.sources import CollegesaurusCorpus, Source
from chatbot.store import Chunk, SqliteNumpyStore

MIN_KEPT_FRACTION = 0.5  # refuse a rebuild that loses more than half a source's chunks
# An unconverted MDX component: <Name then an attribute or the end of the tag,
# so prose such as "<TOEFL 80" is not mistaken for one.
_LEFTOVER_COMPONENT = re.compile(r"<([A-Z][A-Za-z0-9]*)(?:\s+[A-Za-z][\w-]*\s*=|\s*/?>)")


class GuardError(Exception):
    """A freshly built index looks broken, so it is not swapped in."""


class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class GeminiEmbedder:
    """Gemini embeddings reduced to `dim` dimensions, returned L2-normalized.

    gemini-embedding-001 only normalizes its full 3072-dimension output, so the
    smaller vectors are normalized here.
    """

    def __init__(
        self,
        client: genai.Client,
        model: str = config.GEMINI_EMBED_MODEL,
        dim: int = config.EMBED_DIM,
        batch_size: int = 100,
    ):
        self._client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        values: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            values.extend(self._embed(texts[start : start + self.batch_size], "RETRIEVAL_DOCUMENT"))
        return _normalize(np.array(values, dtype=np.float32).reshape(len(texts), self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(self._embed([text], "RETRIEVAL_QUERY"), dtype=np.float32))[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
        )
        return [e.values for e in result.embeddings]


class Refresher:
    """Keeps an index in step with its sources. `store` is the index to serve."""

    def __init__(
        self,
        sources: list[Source],
        embedder: Embedder,
        store: SqliteNumpyStore | None,
        *,
        clock=time.time,
        stale_after: float = config.STALE_AFTER_SECONDS,
    ):
        self.sources = list(sources)
        self.embedder = embedder
        # Never serve vectors from another embedding model (e.g. a snapshot baked
        # before GEMINI_EMBED_MODEL changed): wait for the first refresh instead.
        self.store = store if store is None or self._compatible(store) else None
        self.last_error: str | None = None
        self._clock = clock
        self._stale_after = stale_after
        self.last_ok_at = clock()  # a fresh start counts as up to date

    @property
    def stale(self) -> bool:
        """True when polls have been failing for longer than `stale_after` seconds."""
        return self._clock() - self.last_ok_at > self._stale_after

    def refresh_once(self) -> bool:
        """Rebuild the sources that changed. Returns True if a new index was swapped in."""
        try:
            store = self.store
            if store is not None and not self._compatible(store):
                store = None  # vectors from another model can't be mixed or reused
            changed = False
            for source in self.sources:
                fingerprint = source.fingerprint()
                key = f"source:{source.name}"
                if store is not None and store.meta.get(key) == fingerprint:
                    continue
                chunks = [c for doc in source.load() for c in chunk_document(doc)]
                _check(source.name, chunks, store)
                base = store or SqliteNumpyStore.empty(self.embedder.dim)
                store = base.replace_source(
                    source.name,
                    chunks,
                    self._vectors(chunks, store),
                    meta=base.meta | {key: fingerprint},
                )
                changed = True
            if changed:
                self.store = SqliteNumpyStore(store.chunks, store.vectors, self._meta(store))
            self.last_ok_at = self._clock()
            self.last_error = None
            return changed
        except Exception as exc:  # keep serving the old index; /healthz reports the error
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def _compatible(self, store: SqliteNumpyStore) -> bool:
        return store.meta.get("model") == self.embedder.model and store.meta.get("dim") == str(
            self.embedder.dim
        )

    def _vectors(self, chunks: list[Chunk], previous: SqliteNumpyStore | None) -> np.ndarray:
        """Reuse the vector of any chunk whose text is unchanged; embed the rest."""
        vectors = np.zeros((len(chunks), self.embedder.dim), dtype=np.float32)
        missing: list[int] = []
        for i, chunk in enumerate(chunks):
            known = previous.vector_for_text(chunk.text) if previous is not None else None
            if known is None:
                missing.append(i)
            else:
                vectors[i] = known
        if missing:
            vectors[missing] = self.embedder.embed_documents([chunks[i].text for i in missing])
        return vectors

    def _meta(self, store: SqliteNumpyStore) -> dict[str, str]:
        digest = hashlib.sha256()
        for chunk in store.chunks:
            digest.update(f"{chunk.uid}\0{chunk.text}\0".encode())
        return store.meta | {
            "model": self.embedder.model,
            "dim": str(self.embedder.dim),
            "chunks": str(len(store.chunks)),
            "content_sha": digest.hexdigest()[:16],
            "build_id": " ".join(
                f"{s.name}={store.meta.get(f'source:{s.name}', '')}" for s in self.sources
            ),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }


def _check(source: str, chunks: list[Chunk], previous: SqliteNumpyStore | None) -> None:
    if not chunks:
        raise GuardError(f"{source}: the new build has no chunks")
    if previous is not None:
        old = sum(1 for c in previous.chunks if c.source == source)
        if old and len(chunks) < old * MIN_KEPT_FRACTION:
            raise GuardError(f"{source}: {len(chunks)} chunks, down from {old}")
    for chunk in chunks:
        leftover = _LEFTOVER_COMPONENT.search(chunk.text)
        if leftover:
            raise GuardError(
                f"{source}: unconverted <{leftover.group(1)}> component in {chunk.uid}"
            )


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def main(argv: list[str] | None = None, *, embedder: Embedder | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chatbot.ingest", description="Build the chatbot's search index."
    )
    parser.add_argument("--out", type=Path, default=config.INDEX_PATH, help="index file to write")
    parser.add_argument(
        "--corpus",
        action="append",
        metavar="LOCALE=PATH_OR_URL",
        help="read a corpus.json instead of the live site (repeat per locale)",
    )
    parser.add_argument("--query", help="after building, print the best matches for this")
    parser.add_argument("--locale", default="en", help="search locale for --query")
    args = parser.parse_args(argv)

    if args.corpus:
        sources: list[Source] = [
            CollegesaurusCorpus(dict(item.split("=", 1) for item in args.corpus))
        ]
    else:
        sources = config.sources()
    embedder = embedder or GeminiEmbedder(config.gemini_client())
    previous = SqliteNumpyStore.load(args.out) if args.out.exists() else None
    refresher = Refresher(sources, embedder, previous)

    changed = refresher.refresh_once()
    if refresher.last_error:
        print(f"Index not built: {refresher.last_error}", file=sys.stderr)
        return 1
    meta = refresher.store.meta
    if changed:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        refresher.store.save(args.out)
        print(f"Indexed {meta['chunks']} chunks ({meta['build_id']}) -> {args.out}")
    else:
        print(f"{args.out} is already up to date ({meta['build_id']})")

    if args.query:
        hits = refresher.store.search(embedder.embed_query(args.query), k=5, locale=args.locale)
        print("Top matches:")
        for hit in hits:
            print(f"  {hit.score:.3f}  {hit.chunk.text.splitlines()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
