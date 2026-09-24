"""Shared test doubles. Only external services (Gemini, Supabase, HTTP) are faked."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

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
