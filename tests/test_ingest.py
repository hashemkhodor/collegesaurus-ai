import json

import numpy as np
import pytest
from conftest import local_source
from google.genai import types

from chatbot.ingest import GeminiEmbedder, Refresher, main
from chatbot.sources import Document
from chatbot.store import SqliteNumpyStore

ALL_DOCS = {
    ("ar", "scholarship/fulbright"),
    ("ar", "university/aub"),
    ("en", "scholarship/fulbright"),
    ("en", "university/aub"),
    ("en", "university/lau"),
}


def edit_corpus(directory, locale, change):
    path = directory / f"corpus.{locale}.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))
    change(corpus)
    path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")


def edit_body(directory, locale, slug, transform):
    def change(corpus):
        for doc in corpus["docs"]:
            if doc["slug"] == slug:
                doc["body"] = transform(doc["body"])

    edit_corpus(directory, locale, change)


def raise_aub_tuition(directory):
    edit_body(directory, "en", "aub", lambda b: b.replace("$1,000 per credit", "$1,100 per credit"))


class StaticSource:
    types = {"story": "Student stories."}

    def __init__(self, name, docs):
        self.name = name
        self._docs = docs

    def fingerprint(self):
        return "static"

    def load(self):
        return self._docs


def test_first_refresh_indexes_every_document(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)

    assert refresher.refresh_once() is True
    store = refresher.store
    assert {(c.locale, c.doc_id) for c in store.chunks} == ALL_DOCS
    assert sorted(embedder.embedded) == sorted(c.text for c in store.chunks)


def test_unchanged_source_is_not_rebuilt_or_re_embedded(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    refresher.refresh_once()
    first, embedded = refresher.store, len(embedder.embedded)

    assert refresher.refresh_once() is False
    assert refresher.store is first
    assert len(embedder.embedded) == embedded


def test_changed_page_re_embeds_only_its_changed_chunk(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    refresher.refresh_once()
    before = len(embedder.embedded)

    raise_aub_tuition(corpus_dir)

    assert refresher.refresh_once() is True
    new_texts = embedder.embedded[before:]
    assert len(new_texts) == 1
    assert "$1,100 per credit" in new_texts[0]
    texts = [c.text for c in refresher.store.chunks]
    assert any("$1,100 per credit" in t for t in texts)
    assert not any("$1,000 per credit" in t for t in texts)


def test_vectors_from_another_embedding_model_are_not_reused(corpus_dir, embedder):
    built = Refresher([local_source(corpus_dir)], embedder, store=None)
    built.refresh_once()
    old = built.store
    other_model = SqliteNumpyStore(
        old.chunks, old.vectors, old.meta | {"model": "other-model", "source:collegesaurus": "x"}
    )
    before = len(embedder.embedded)

    Refresher([local_source(corpus_dir)], embedder, store=other_model).refresh_once()

    assert len(embedder.embedded) - before == len(old.chunks)


def test_refresh_that_loses_most_chunks_is_refused_and_the_old_index_kept(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    refresher.refresh_once()
    kept = refresher.store

    edit_corpus(corpus_dir, "en", lambda c: c.update(docs=c["docs"][2:]))  # only fulbright
    edit_corpus(corpus_dir, "ar", lambda c: c.update(docs=[]))

    assert refresher.refresh_once() is False
    assert refresher.store is kept
    assert "GuardError" in refresher.last_error


def test_leftover_mdx_components_are_refused(corpus_dir, embedder):
    edit_body(corpus_dir, "en", "aub", lambda b: b + "\n<MajorsTable rows={[{program: 'X'}]} />\n")
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)

    assert refresher.refresh_once() is False
    assert refresher.store is None
    assert "MajorsTable" in refresher.last_error


def test_a_failing_source_keeps_the_old_index_and_records_the_error(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    refresher.refresh_once()
    kept = refresher.store

    (corpus_dir / "corpus.ar.json").unlink()

    assert refresher.refresh_once() is False
    assert refresher.store is kept
    assert "CorpusError" in refresher.last_error


def test_a_later_successful_refresh_clears_the_error(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    backup = (corpus_dir / "corpus.ar.json").read_bytes()
    (corpus_dir / "corpus.ar.json").unlink()
    refresher.refresh_once()

    (corpus_dir / "corpus.ar.json").write_bytes(backup)

    assert refresher.refresh_once() is True
    assert refresher.last_error is None


def test_index_turns_stale_only_after_polls_keep_failing(corpus_dir, embedder):
    now = [1000.0]
    refresher = Refresher(
        [local_source(corpus_dir)], embedder, None, clock=lambda: now[0], stale_after=600
    )
    refresher.refresh_once()
    (corpus_dir / "corpus.ar.json").unlink()

    now[0] = 1500.0
    refresher.refresh_once()
    assert refresher.stale is False

    now[0] = 1601.0
    refresher.refresh_once()
    assert refresher.stale is True


def test_build_details_are_recorded_in_the_index_meta(corpus_dir, embedder):
    source = local_source(corpus_dir)
    refresher = Refresher([source], embedder, store=None)
    refresher.refresh_once()
    meta = refresher.store.meta

    assert (meta["model"], meta["dim"], meta["chunks"]) == ("fake-embedding", "64", "10")
    assert meta["build_id"] == f"collegesaurus={source.fingerprint()}"
    content_sha = meta["content_sha"]

    raise_aub_tuition(corpus_dir)
    refresher.refresh_once()

    assert refresher.store.meta["content_sha"] != content_sha


def test_other_sources_keep_their_chunks_when_one_source_changes(corpus_dir, embedder):
    story = Document(
        source="stories",
        doc_id="story/first-year",
        type="story",
        locale="en",
        serves=("en",),
        title="My first year at AUB",
        url="https://collegesaurus.org/stories/first-year",
        body="## Story\n\nMy first year was hard but worth it.",
    )
    refresher = Refresher(
        [local_source(corpus_dir), StaticSource("stories", [story])], embedder, store=None
    )
    refresher.refresh_once()
    before = len(embedder.embedded)

    raise_aub_tuition(corpus_dir)
    refresher.refresh_once()

    assert [c.doc_id for c in refresher.store.chunks if c.source == "stories"] == [
        "story/first-year"
    ]
    assert len(embedder.embedded) - before == 1


class FakeEmbedModels:
    """Stands in for client.models of the google-genai SDK."""

    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append((model, list(contents), config))
        values = [3.0, 4.0] + [0.0] * (config.output_dimensionality - 2)
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=values) for _ in contents]
        )


class FakeGenaiClient:
    def __init__(self):
        self.models = FakeEmbedModels()


def test_gemini_embedder_batches_documents_and_returns_unit_vectors():
    client = FakeGenaiClient()
    embedder = GeminiEmbedder(client, model="gemini-embedding-001", dim=768, batch_size=100)

    vectors = embedder.embed_documents([f"text {i}" for i in range(250)])

    calls = client.models.calls
    assert [len(contents) for _, contents, _ in calls] == [100, 100, 50]
    assert {model for model, _, _ in calls} == {"gemini-embedding-001"}
    assert {c.task_type for _, _, c in calls} == {"RETRIEVAL_DOCUMENT"}
    assert {c.output_dimensionality for _, _, c in calls} == {768}
    assert vectors.shape == (250, 768)
    assert vectors[0][:2].tolist() == pytest.approx([0.6, 0.8])


def test_gemini_embedder_uses_the_query_task_for_questions():
    client = FakeGenaiClient()

    vector = GeminiEmbedder(client, dim=768).embed_query("AUB tuition")

    [(_, contents, config)] = client.models.calls
    assert contents == ["AUB tuition"]
    assert config.task_type == "RETRIEVAL_QUERY"
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def corpus_args(directory):
    return [
        "--corpus",
        f"en={directory / 'corpus.en.json'}",
        "--corpus",
        f"ar={directory / 'corpus.ar.json'}",
    ]


def test_cli_builds_and_saves_an_index_from_local_corpora(corpus_dir, embedder, tmp_path):
    out = tmp_path / "data" / "index.db"

    assert main([*corpus_args(corpus_dir), "--out", str(out)], embedder=embedder) == 0

    assert len(SqliteNumpyStore.load(out).chunks) == 10


def test_cli_query_prints_the_best_matching_section(corpus_dir, embedder, tmp_path, capsys):
    out = tmp_path / "index.db"
    args = [*corpus_args(corpus_dir), "--out", str(out), "--query", "AUB tuition per credit"]

    main(args, embedder=embedder)

    results = capsys.readouterr().out.split("Top matches:\n", 1)[1]
    assert "Tuition (AY 2026-2027)" in results.splitlines()[0]


def test_cli_exits_non_zero_when_the_index_cannot_be_built(corpus_dir, embedder, tmp_path):
    (corpus_dir / "corpus.ar.json").unlink()

    code = main([*corpus_args(corpus_dir), "--out", str(tmp_path / "index.db")], embedder=embedder)

    assert code == 1
    assert not (tmp_path / "index.db").exists()


def test_a_snapshot_from_another_embedding_model_is_never_served(corpus_dir, embedder):
    built = Refresher([local_source(corpus_dir)], embedder, store=None)
    built.refresh_once()
    old = built.store
    other_model = SqliteNumpyStore(old.chunks, old.vectors, old.meta | {"model": "other-model"})

    assert Refresher([local_source(corpus_dir)], embedder, other_model).store is None


def test_angle_brackets_in_prose_do_not_trip_the_component_guard(corpus_dir, embedder):
    edit_body(corpus_dir, "en", "aub", lambda b: b + "\nTransfer applicants need <TOEFL 80.\n")
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)

    assert refresher.refresh_once() is True
