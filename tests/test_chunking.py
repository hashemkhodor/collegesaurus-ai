import json
from pathlib import Path

from chatbot.chunking import chunk_document
from chatbot.sources import Document

FIXTURES = Path(__file__).parent / "fixtures"
AUB_TITLE = "AUB — American University of Beirut"
TABLE_HEADER = "| Program | Degree | Department | Credits | Years |\n|---|---|---|---|---|"


def fixture_body(slug: str) -> str:
    corpus = json.loads((FIXTURES / "corpus.en.json").read_text(encoding="utf-8"))
    return next(d["body"] for d in corpus["docs"] if d["slug"] == slug)


def make_doc(body: str, title: str = AUB_TITLE, year: str | None = "2026-2027") -> Document:
    return Document(
        source="collegesaurus",
        doc_id="university/aub",
        type="university",
        locale="en",
        serves=("en", "ar"),
        title=title,
        url="https://collegesaurus.org/universities/aub",
        body=body,
        metadata={"content_year": year} if year else {},
    )


def body_of(chunk) -> str:
    return chunk.text.split("\n\n", 1)[1]


def test_small_sections_become_one_chunk_each_with_a_breadcrumb_first_line():
    chunks = chunk_document(make_doc(fixture_body("aub")))

    assert [c.text.splitlines()[0] for c in chunks] == [
        f"{AUB_TITLE} › Faculty [2026-2027]",
        f"{AUB_TITLE} › Tuition (AY 2026-2027) [2026-2027]",
        f"{AUB_TITLE} › Contacts [2026-2027]",
    ]
    assert "| Civil Engineering | BE |" in chunks[0].text
    assert "**$1,000 per credit**" in chunks[1].text


def test_page_title_heading_never_becomes_a_chunk_of_its_own():
    chunks = chunk_document(make_doc(fixture_body("aub")))

    assert not any("# American University of Beirut (AUB)" in c.text for c in chunks)


def test_text_before_the_first_section_is_kept_under_the_page_breadcrumb():
    lau = make_doc(
        fixture_body("lau"), title="LAU — Lebanese American University", year="2025-2026"
    )

    first = chunk_document(lau)[0]

    assert first.text.splitlines()[0] == "LAU — Lebanese American University [2025-2026]"
    assert "Not yet updated for 2026-2027" in first.text


def test_oversized_section_splits_by_subsection_with_link_free_breadcrumbs():
    rows = "\n".join(
        f"| Program {i:02d} | BE | Department of Engineering | 150 | 4 |" for i in range(30)
    )
    body = (
        "## Faculty\n\nAUB offers ~46 undergraduate programs.\n\n"
        "### Maroun Semaan Faculty of Engineering & Architecture ([MSFEA](https://aub.edu.lb/msfea))\n\n"
        f"{TABLE_HEADER}\n{rows}\n\n"
        "### Faculty of Arts and Sciences ([FAS](https://aub.edu.lb/fas))\n\n"
        f"{TABLE_HEADER}\n{rows}\n"
    )

    first_lines = {c.text.splitlines()[0] for c in chunk_document(make_doc(body))}

    msfea = "Maroun Semaan Faculty of Engineering & Architecture (MSFEA)"
    assert first_lines == {
        f"{AUB_TITLE} › Faculty [2026-2027]",
        f"{AUB_TITLE} › Faculty › {msfea} [2026-2027]",
        f"{AUB_TITLE} › Faculty › Faculty of Arts and Sciences (FAS) [2026-2027]",
    }


def test_long_table_splits_by_rows_repeating_the_header_without_losing_rows():
    rows = [f"| Program {i:02d} | BE | Department of Engineering | 150 | 4 |" for i in range(60)]
    body = "## Faculty\n\n### Engineering\n\n" + TABLE_HEADER + "\n" + "\n".join(rows) + "\n"

    chunks = chunk_document(make_doc(body))

    assert len(chunks) > 1
    assert all(TABLE_HEADER in c.text for c in chunks)
    found = [
        line
        for c in chunks
        for line in c.text.splitlines()
        if line.startswith("| Program ") and "Degree" not in line
    ]
    assert found == rows


def test_no_chunk_body_exceeds_the_size_cap_and_no_sentence_is_lost():
    sentences = [f"Sentence {i} explains one admission rule in detail." for i in range(150)]
    body = "## Requirements\n\n" + " ".join(sentences) + "\n"

    chunks = chunk_document(make_doc(body), max_chars=1200)

    assert all(len(body_of(c)) <= 1200 for c in chunks)
    assert " ".join(body_of(c) for c in chunks).split(". ") == " ".join(sentences).split(". ")


def test_long_list_is_split_between_items():
    items = [f"- Required document number {i}: certified copy" for i in range(80)]
    body = "## Application\n\n" + "\n".join(items) + "\n"

    chunks = chunk_document(make_doc(body), max_chars=1200)

    assert len(chunks) > 1
    assert [line for c in chunks for line in body_of(c).splitlines()] == items


def test_sections_without_content_are_dropped():
    chunks = chunk_document(make_doc("## Empty\n\n## Contacts\n\nCall 01-350000.\n"))

    assert [c.section for c in chunks] == ["Contacts"]


def test_chunks_carry_stable_ids_and_the_document_metadata():
    chunks = chunk_document(make_doc(fixture_body("aub")))

    assert [c.uid for c in chunks] == [
        "collegesaurus:en:university/aub:0",
        "collegesaurus:en:university/aub:1",
        "collegesaurus:en:university/aub:2",
    ]
    tuition = chunks[1]
    assert tuition.section == "Tuition (AY 2026-2027)"
    assert tuition.url == "https://collegesaurus.org/universities/aub"
    assert tuition.title == AUB_TITLE
    assert (tuition.type, tuition.locale, tuition.serves) == ("university", "en", ("en", "ar"))
    assert tuition.metadata == {"content_year": "2026-2027"}


def test_breadcrumb_has_no_year_tag_when_the_page_has_no_content_year():
    chunks = chunk_document(make_doc("## Contacts\n\nCall 01-350000.\n", year=None))

    assert chunks[0].text.splitlines()[0] == f"{AUB_TITLE} › Contacts"
