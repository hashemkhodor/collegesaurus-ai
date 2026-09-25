import json
import shutil
from pathlib import Path

import httpx
import pytest

from chatbot.sources import CollegesaurusCorpus, CorpusError

FIXTURES = Path(__file__).parent / "fixtures"
SITE = "https://collegesaurus.org"


def local_corpus(directory: Path = FIXTURES) -> CollegesaurusCorpus:
    return CollegesaurusCorpus(
        {"en": str(directory / "corpus.en.json"), "ar": str(directory / "corpus.ar.json")}
    )


def fixture_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def http_corpus(seen: list[httpx.Request], status: int = 200) -> CollegesaurusCorpus:
    routes = {
        "/chatbot/version.json": fixture_json("version.json"),
        "/chatbot/corpus.json": fixture_json("corpus.en.json"),
        "/ar/chatbot/corpus.json": fixture_json("corpus.ar.json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200 or request.url.path not in routes:
            return httpx.Response(status if status != 200 else 404)
        return httpx.Response(200, json=routes[request.url.path])

    return CollegesaurusCorpus(
        {"en": f"{SITE}/chatbot/corpus.json", "ar": f"{SITE}/ar/chatbot/corpus.json"},
        version=f"{SITE}/chatbot/version.json",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_load_keeps_english_docs_and_only_real_arabic_translations():
    docs = local_corpus().load()

    assert sorted((d.locale, d.doc_id) for d in docs) == [
        ("ar", "scholarship/fulbright"),
        ("ar", "university/aub"),
        ("en", "scholarship/fulbright"),
        ("en", "university/aub"),
        ("en", "university/lau"),
    ]


def test_english_page_without_arabic_translation_also_serves_arabic_searches():
    by_key = {(d.locale, d.doc_id): d for d in local_corpus().load()}

    assert by_key[("en", "university/lau")].serves == ("en", "ar")
    assert by_key[("en", "university/aub")].serves == ("en",)
    assert by_key[("ar", "university/aub")].serves == ("ar",)


def test_document_fields_come_from_the_corpus_entry():
    aub = next(
        d for d in local_corpus().load() if d.locale == "en" and d.doc_id == "university/aub"
    )

    assert aub.source == "collegesaurus"
    assert aub.type == "university"
    assert aub.title == "AUB — American University of Beirut"
    assert aub.url == "https://collegesaurus.org/universities/aub"
    assert aub.metadata == {
        "content_year": "2026-2027",
        "apply_url": "https://join.aub.edu.lb/apply/",
    }
    assert aub.body.startswith("# American University of Beirut (AUB)")


def test_docs_with_an_empty_body_are_skipped(tmp_path):
    shutil.copy(FIXTURES / "corpus.ar.json", tmp_path / "corpus.ar.json")
    corpus = fixture_json("corpus.en.json")
    corpus["docs"][0]["body"] = "   \n"
    (tmp_path / "corpus.en.json").write_text(json.dumps(corpus), encoding="utf-8")

    docs = local_corpus(tmp_path).load()

    assert ("en", "university/aub") not in {(d.locale, d.doc_id) for d in docs}


def test_rejects_a_corpus_with_an_unknown_schema(tmp_path):
    shutil.copy(FIXTURES / "corpus.ar.json", tmp_path / "corpus.ar.json")
    corpus = fixture_json("corpus.en.json") | {"schema": 2}
    (tmp_path / "corpus.en.json").write_text(json.dumps(corpus), encoding="utf-8")

    with pytest.raises(CorpusError, match="schema"):
        local_corpus(tmp_path).load()


def test_rejects_a_corpus_published_under_the_wrong_locale(tmp_path):
    shutil.copy(FIXTURES / "corpus.en.json", tmp_path / "corpus.ar.json")
    shutil.copy(FIXTURES / "corpus.en.json", tmp_path / "corpus.en.json")

    with pytest.raises(CorpusError, match="locale"):
        local_corpus(tmp_path).load()


def test_fingerprint_of_local_files_changes_only_when_a_file_changes(tmp_path):
    for name in ("corpus.en.json", "corpus.ar.json"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    source = local_corpus(tmp_path)
    before = source.fingerprint()

    assert source.fingerprint() == before
    corpus = fixture_json("corpus.ar.json")
    corpus["docs"][0]["body"] += "\nNew paragraph."
    (tmp_path / "corpus.ar.json").write_text(json.dumps(corpus), encoding="utf-8")
    assert source.fingerprint() != before


def test_fingerprint_reads_version_json_past_the_cdn_cache():
    seen: list[httpx.Request] = []

    fingerprint = http_corpus(seen).fingerprint()

    assert fingerprint == "drivefp-1|abc1234"
    assert [r.url.path for r in seen] == ["/chatbot/version.json"]
    assert "t" in seen[0].url.params


def test_load_fetches_every_locale_over_http_past_the_cdn_cache():
    seen: list[httpx.Request] = []

    docs = http_corpus(seen).load()

    assert len(docs) == 5
    assert sorted(r.url.path for r in seen) == ["/ar/chatbot/corpus.json", "/chatbot/corpus.json"]
    assert all("t" in r.url.params for r in seen)


def test_http_failure_is_reported_as_a_corpus_error():
    with pytest.raises(CorpusError, match="404"):
        http_corpus([], status=404).load()
