"""The search index: chunks of text with their embedding vectors.

At a few thousand chunks a vector-database engine adds nothing, so the index
is a plain SQLite file (standard library) holding each chunk's text, metadata
and vector, loaded into a numpy matrix for exact cosine search. It is
immutable: a refresh builds a new store and swaps it in.

VectorStore is the seam for growing out of this: a much larger step-two
source could swap in sqlite-vec, pgvector or Chroma without touching sources,
chunking or tools.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Chunk:
    uid: str  # "<source>:<locale>:<doc_id>:<n>"
    source: str
    doc_id: str
    type: str
    locale: str
    serves: tuple[str, ...]  # search locales this chunk answers for
    title: str  # page title
    url: str  # page URL (not a section anchor; anchors change every year)
    section: str  # "Faculty › MSFEA", "" for text before the first section
    text: str  # breadcrumb line + markdown body; embedded and shown to the model
    metadata: dict[str, str] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity to the query


@dataclass(frozen=True)
class Page:
    doc_id: str
    type: str
    title: str
    url: str


class VectorStore(Protocol):
    chunks: list[Chunk]
    meta: dict[str, str]

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        locale: str | None = None,
        types: Collection[str] | None = None,
    ) -> list[Hit]: ...

    def pages(
        self, *, locale: str | None = None, types: Collection[str] | None = None
    ) -> list[Page]: ...


_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunks (
    uid TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    type TEXT NOT NULL,
    locale TEXT NOT NULL,
    serves TEXT NOT NULL,      -- JSON list of search locales
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,    -- JSON object
    vector BLOB NOT NULL       -- float32, unit length
);
"""


class SqliteNumpyStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict[str, str]):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but vectors of shape {vectors.shape}")
        self.chunks = list(chunks)
        self.vectors = vectors
        self.meta = dict(meta)
        self._types = np.array([c.type for c in self.chunks], dtype=object)
        self._by_text = {c.text: i for i, c in enumerate(self.chunks)}

    @classmethod
    def empty(cls, dim: int, meta: dict[str, str] | None = None) -> SqliteNumpyStore:
        return cls([], np.zeros((0, dim), dtype=np.float32), meta or {})

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        locale: str | None = None,
        types: Collection[str] | None = None,
    ) -> list[Hit]:
        mask = np.ones(len(self.chunks), dtype=bool)
        if locale:
            mask &= np.array([locale in c.serves for c in self.chunks], dtype=bool)
        if types:
            mask &= np.isin(self._types, list(types))
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        # Score every row, then pick: indexing the matrix first would copy it for
        # every search, which adds up when many chats search at once.
        scores = (self.vectors @ np.asarray(query, dtype=np.float32))[candidates]
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[candidates[i]], float(scores[i])) for i in order]

    def pages(
        self, *, locale: str | None = None, types: Collection[str] | None = None
    ) -> list[Page]:
        seen: dict[str, Page] = {}
        for c in self.chunks:
            if (locale and locale not in c.serves) or (types and c.type not in types):
                continue
            seen.setdefault(c.doc_id, Page(c.doc_id, c.type, c.title, c.url))
        return sorted(seen.values(), key=lambda p: (p.type, p.title.casefold()))

    def vector_for_text(self, text: str) -> np.ndarray | None:
        i = self._by_text.get(text)
        return None if i is None else self.vectors[i]

    def replace_source(
        self,
        source: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
        meta: dict[str, str] | None = None,
    ) -> SqliteNumpyStore:
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        return SqliteNumpyStore(
            [self.chunks[i] for i in keep] + list(chunks),
            np.concatenate([self.vectors[keep], np.asarray(vectors, dtype=np.float32)]),
            self.meta if meta is None else meta,
        )

    def save(self, path: Path) -> None:
        """Write the index to `path`, replacing any existing file atomically."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.unlink(missing_ok=True)
        con = sqlite3.connect(tmp)
        try:
            con.executescript(_SCHEMA)
            con.executemany("INSERT INTO meta VALUES (?, ?)", self.meta.items())
            con.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.uid,
                        c.source,
                        c.doc_id,
                        c.type,
                        c.locale,
                        json.dumps(list(c.serves)),
                        c.title,
                        c.url,
                        c.section,
                        c.text,
                        json.dumps(c.metadata, ensure_ascii=False),
                        v.tobytes(),
                    )
                    for c, v in zip(self.chunks, self.vectors, strict=True)
                ],
            )
            con.commit()
        finally:
            con.close()
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> SqliteNumpyStore:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            meta = dict(con.execute("SELECT key, value FROM meta"))
            rows = con.execute(
                "SELECT uid, source, doc_id, type, locale, serves, title, url, section, text,"
                " metadata, vector FROM chunks ORDER BY rowid"
            ).fetchall()
        finally:
            con.close()
        chunks = [
            Chunk(
                uid=r[0],
                source=r[1],
                doc_id=r[2],
                type=r[3],
                locale=r[4],
                serves=tuple(json.loads(r[5])),
                title=r[6],
                url=r[7],
                section=r[8],
                text=r[9],
                metadata=json.loads(r[10]),
            )
            for r in rows
        ]
        if not rows:
            return cls.empty(int(meta.get("dim", 0)), meta)
        vectors = np.stack([np.frombuffer(r[11], dtype=np.float32) for r in rows])
        return cls(chunks, vectors, meta)
