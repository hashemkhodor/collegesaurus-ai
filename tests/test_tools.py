import json

import pytest
from conftest import local_source

from chatbot.ingest import Refresher
from chatbot.sources import CollegesaurusCorpus
from chatbot.tools import ToolRunner, question_locale, tool_declarations

AUB_URL = "https://collegesaurus.org/universities/aub"
LAU_URL = "https://collegesaurus.org/universities/lau"
STORY_URL = "https://collegesaurus.org/stories/scholarship-awardees/abdelhamid-stipendium"


@pytest.fixture
def store(corpus_dir, embedder):
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    refresher.refresh_once()
    return refresher.store


def runner(store, embedder, locale="en", top_k=15):
    return ToolRunner(
        store, embedder.embed_query, CollegesaurusCorpus.types, locale=locale, top_k=top_k
    )


@pytest.mark.parametrize(
    ("text", "locale"),
    [
        ("What is AUB's tuition?", "en"),
        ("Quels sont les frais de scolarité à l'USJ ?", "en"),
        ("ما هي أقساط الجامعة الأميركية؟", "ar"),
        ("ما هي أقساط AUB؟", "ar"),
    ],
)
def test_question_locale_picks_arabic_only_for_mostly_arabic_questions(text, locale):
    assert question_locale(text) == locale


def test_declarations_offer_search_and_list_pages_with_the_registered_types():
    search, list_pages = tool_declarations({"university": "Universities.", "story": "Stories."})

    assert search.name == "search"
    assert search.parameters.required == ["query"]
    assert search.parameters.properties["type"].enum == ["university", "story"]
    assert list_pages.name == "list_pages"
    assert list_pages.parameters.properties["type"].enum == ["university", "story"]


def test_search_returns_numbered_passages_with_breadcrumb_and_source(store, embedder):
    result = runner(store, embedder, top_k=1).run("search", {"query": "AUB tuition per credit"})

    assert result.startswith(
        "[1] AUB — American University of Beirut › Tuition (AY 2026-2027) [2026-2027]\n"
        f"Source: {AUB_URL}\n"
        "Tuition is **$1,000 per credit**"
    )


def test_arabic_search_finds_arabic_pages_and_untranslated_english_ones(store, embedder):
    tools = runner(store, embedder, locale="ar")

    tools.run("search", {"query": "AUB LAU Fulbright"})

    assert {(h.chunk.locale, h.chunk.doc_id) for h in tools.hits} == {
        ("ar", "university/aub"),
        ("ar", "scholarship/fulbright"),
        ("en", "university/lau"),
    }


def test_search_can_be_limited_to_one_type(store, embedder):
    tools = runner(store, embedder)

    tools.run("search", {"query": "application deadline", "type": "scholarship"})

    assert {h.chunk.type for h in tools.hits} == {"scholarship"}


def add_story(corpus_dir):
    """A Stories post as the site publishes it: English only, so the Arabic
    corpus carries a fallback copy of the English body."""
    story = {
        "type": "story",
        "slug": "scholarship-awardees/abdelhamid-stipendium",
        "title": "Stipendium Hungaricum, From Lebanon",
        "url": STORY_URL,
        "content_locale": "en",
        "content_year": None,
        "apply_url": None,
        "body": (
            "*By Abdelhamid Khaled, 17 May 2026.*\n\n"
            "I first heard about Stipendium Hungaricum from a friend. In Lebanon the "
            "application runs through the Ministry of Education."
        ),
    }
    for name in ("corpus.en.json", "corpus.ar.json"):
        path = corpus_dir / name
        corpus = json.loads(path.read_text(encoding="utf-8"))
        corpus["docs"].append(story)
        path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")


def test_stories_can_be_searched_and_listed_in_both_languages(corpus_dir, embedder):
    add_story(corpus_dir)
    refresher = Refresher([local_source(corpus_dir)], embedder, store=None)
    refresher.refresh_once()

    for locale in ("en", "ar"):
        tools = runner(refresher.store, embedder, locale=locale)
        found = tools.run("search", {"query": "Stipendium Hungaricum experience", "type": "story"})
        listed = tools.run("list_pages", {"type": "story"})

        assert found.startswith("[1] Stipendium Hungaricum, From Lebanon\n"), found
        assert {h.chunk.type for h in tools.hits} == {"story"}
        assert listed == f"- Stipendium Hungaricum, From Lebanon ({STORY_URL})"


def test_bad_tool_calls_get_an_error_the_model_can_recover_from(store, embedder):
    tools = runner(store, embedder)

    assert tools.run("search", {"query": "x", "type": "museum"}).startswith("Error: unknown type")
    assert tools.run("search", {}).startswith("Error:")
    assert tools.run("delete_everything", {}).startswith("Error: unknown tool")


def test_list_pages_lists_each_page_once_for_the_locale(store, embedder):
    result = runner(store, embedder).run("list_pages", {"type": "university"})

    assert result.splitlines() == [
        f"- AUB — American University of Beirut ({AUB_URL})",
        f"- LAU — Lebanese American University ({LAU_URL})",
    ]


def test_cited_sources_are_the_retrieved_pages_the_answer_links_to(store, embedder):
    tools = runner(store, embedder)
    tools.run("search", {"query": "AUB LAU tuition application"})
    answer = f"See [AUB]({AUB_URL}) and again [AUB]({AUB_URL}), plus [x](https://example.com)."

    assert tools.cited_sources(answer) == [
        {"title": "AUB — American University of Beirut", "url": AUB_URL}
    ]
    assert tools.cited_sources("No links here.") == []


def test_top_score_is_the_best_retrieval_match_of_the_turn(store, embedder):
    tools = runner(store, embedder)
    assert tools.top_score is None

    tools.run("search", {"query": "AUB tuition per credit"})

    assert tools.top_score == max(h.score for h in tools.hits)
    assert 0 < tools.top_score <= 1
