import numpy as np
import pytest

from chatbot.store import Chunk, SqliteNumpyStore


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def chunk(uid: str, **overrides) -> Chunk:
    fields = dict(
        uid=uid,
        source="collegesaurus",
        doc_id=f"university/{uid}",
        type="university",
        locale="en",
        serves=("en",),
        title=f"Title {uid}",
        url=f"https://collegesaurus.org/universities/{uid}",
        section="Tuition",
        text=f"text of {uid}",
        metadata={"content_year": "2026-2027"},
    )
    return Chunk(**(fields | overrides))


def store_of(chunks: list[Chunk], vectors: list[np.ndarray], meta=None) -> SqliteNumpyStore:
    return SqliteNumpyStore(chunks, np.stack(vectors), meta or {})


def test_search_ranks_chunks_by_cosine_similarity():
    store = store_of([chunk("a"), chunk("b"), chunk("c")], [unit(1, 0), unit(0, 1), unit(0.6, 0.8)])

    hits = store.search(unit(1, 0), k=3)

    assert [h.chunk.uid for h in hits] == ["a", "c", "b"]
    assert [round(h.score, 2) for h in hits] == [1.0, 0.6, 0.0]


def test_search_returns_at_most_k_hits():
    store = store_of([chunk("a"), chunk("b"), chunk("c")], [unit(1, 0), unit(0, 1), unit(0.6, 0.8)])

    assert [h.chunk.uid for h in store.search(unit(1, 0), k=2)] == ["a", "c"]


def test_search_can_be_limited_to_some_types():
    store = store_of(
        [chunk("aub"), chunk("fulbright", type="scholarship", doc_id="scholarship/fulbright")],
        [unit(1, 0), unit(0, 1)],
    )

    hits = store.search(unit(1, 0), k=5, types={"scholarship"})

    assert [h.chunk.uid for h in hits] == ["fulbright"]


def test_search_only_returns_chunks_that_serve_the_requested_locale():
    store = store_of(
        [
            chunk("aub-en"),
            chunk("aub-ar", locale="ar", serves=("ar",)),
            chunk("lau-en", serves=("en", "ar")),
        ],
        [unit(1, 0), unit(1, 0.1), unit(1, 0.2)],
    )

    assert {h.chunk.uid for h in store.search(unit(1, 0), k=5, locale="ar")} == {"aub-ar", "lau-en"}
    assert {h.chunk.uid for h in store.search(unit(1, 0), k=5, locale="en")} == {"aub-en", "lau-en"}


def test_empty_store_finds_nothing():
    assert SqliteNumpyStore.empty(dim=2).search(unit(1, 0), k=5) == []


def test_pages_lists_each_document_once_for_the_requested_locale():
    store = store_of(
        [
            chunk("aub-0", doc_id="university/aub", title="AUB"),
            chunk("aub-1", doc_id="university/aub", title="AUB"),
            chunk("aub-ar", doc_id="university/aub", title="AUB ar", locale="ar", serves=("ar",)),
            chunk("lau", doc_id="university/lau", title="LAU", serves=("en", "ar")),
            chunk("fb", doc_id="scholarship/fulbright", title="Fulbright", type="scholarship"),
        ],
        [unit(1, 0)] * 5,
    )

    assert [(p.doc_id, p.title) for p in store.pages(locale="ar")] == [
        ("university/aub", "AUB ar"),
        ("university/lau", "LAU"),
    ]
    assert [p.title for p in store.pages(locale="en", types={"scholarship"})] == ["Fulbright"]


def test_saved_index_loads_back_with_the_same_chunks_vectors_and_meta(tmp_path):
    store = store_of(
        [chunk("a", serves=("en", "ar")), chunk("b", metadata={})],
        [unit(1, 0), unit(0.6, 0.8)],
        meta={"build_id": "drivefp-1|abc1234", "model": "gemini-embedding-001"},
    )
    path = tmp_path / "index.db"

    store.save(path)
    loaded = SqliteNumpyStore.load(path)

    assert loaded.chunks == store.chunks
    np.testing.assert_allclose(loaded.vectors, store.vectors)
    assert loaded.meta == {"build_id": "drivefp-1|abc1234", "model": "gemini-embedding-001"}


def test_saving_over_an_existing_index_replaces_its_contents(tmp_path):
    path = tmp_path / "index.db"
    store_of([chunk("old")], [unit(1, 0)]).save(path)

    store_of([chunk("new")], [unit(0, 1)]).save(path)

    assert [c.uid for c in SqliteNumpyStore.load(path).chunks] == ["new"]


def test_replace_source_swaps_one_source_and_keeps_the_others():
    store = store_of(
        [chunk("cs-1"), chunk("story-1", source="stories"), chunk("cs-2")],
        [unit(1, 0), unit(0, 1), unit(0.6, 0.8)],
    )

    new = store.replace_source("collegesaurus", [chunk("cs-new")], np.stack([unit(0.8, 0.6)]))

    assert [c.uid for c in new.chunks] == ["story-1", "cs-new"]
    np.testing.assert_allclose(new.vectors, np.stack([unit(0, 1), unit(0.8, 0.6)]))
    assert [c.uid for c in store.chunks] == ["cs-1", "story-1", "cs-2"]


def test_vector_for_text_returns_the_stored_vector_of_identical_text():
    store = store_of([chunk("a", text="same text"), chunk("b")], [unit(1, 0), unit(0, 1)])

    np.testing.assert_allclose(store.vector_for_text("same text"), unit(1, 0))
    assert store.vector_for_text("never indexed") is None


def test_rejects_vectors_that_do_not_match_the_chunks():
    with pytest.raises(ValueError):
        store_of([chunk("a"), chunk("b")], [unit(1, 0)])
