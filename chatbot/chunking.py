"""Split a Document's markdown into search chunks.

A chunk is one section (H2) of a page; a section that is too long is split
into its subsections (H3), and anything still too long into runs of
paragraphs, list items or table rows (repeating the table header). Every
chunk starts with a breadcrumb line such as

    AUB — American University of Beirut › Tuition (AY 2026-2027) [2026-2027]

so a chunk never loses its page, its section or the academic year it covers.
"""

from __future__ import annotations

import re

from chatbot.sources import Document
from chatbot.store import Chunk

MAX_CHARS = 1200  # per chunk body, not counting the breadcrumb line

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"\*\*|__|`")
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")
_BLANK_LINES = re.compile(r"\n\s*\n")


def chunk_document(doc: Document, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, body in _pieces(doc.body, max_chars):
        for part in [body] if len(body) <= max_chars else _pack(body, max_chars):
            chunks.append(
                Chunk(
                    uid=f"{doc.source}:{doc.locale}:{doc.doc_id}:{len(chunks)}",
                    source=doc.source,
                    doc_id=doc.doc_id,
                    type=doc.type,
                    locale=doc.locale,
                    serves=doc.serves,
                    title=doc.title,
                    url=doc.url,
                    section=" › ".join(_plain(p) for p in path),
                    text=f"{_breadcrumb(doc, path)}\n\n{part}",
                    metadata=dict(doc.metadata),
                )
            )
    return chunks


def _pieces(markdown: str, max_chars: int) -> list[tuple[tuple[str, ...], str]]:
    """(heading path, body) pairs: whole H2 sections, or their H3 parts if too long."""
    intro: list[str] = []
    sections: list[dict] = []  # {"title", "lines", "lead", "subs": [(title, lines)]}
    for line in markdown.replace("\r\n", "\n").split("\n"):
        heading = _HEADING.match(line)
        level = len(heading.group(1)) if heading else 0
        if level == 1:
            continue  # the page title; the breadcrumb already carries it
        if level == 2 or (level == 3 and not sections):
            sections.append({"title": heading.group(2), "lines": [], "lead": [], "subs": []})
            continue
        if sections:
            section = sections[-1]
            section["lines"].append(line)
            if level == 3:
                section["subs"].append((heading.group(2), []))
            elif section["subs"]:
                section["subs"][-1][1].append(line)
            else:
                section["lead"].append(line)
        else:
            intro.append(line)

    pieces: list[tuple[tuple[str, ...], str]] = [((), _text(intro))]
    for section in sections:
        whole = _text(section["lines"])
        if len(whole) <= max_chars or not section["subs"]:
            pieces.append(((section["title"],), whole))
            continue
        pieces.append(((section["title"],), _text(section["lead"])))
        for sub_title, sub_lines in section["subs"]:
            pieces.append(((section["title"], sub_title), _text(sub_lines)))
    return [(path, body) for path, body in pieces if body]


def _pack(body: str, max_chars: int) -> list[str]:
    """Group blocks (paragraphs, tables, lists) into bodies of at most max_chars."""
    out: list[str] = []
    current: list[str] = []
    for block in (b.strip("\n") for b in _BLANK_LINES.split(body) if b.strip()):
        if len(block) > max_chars:
            if current:
                out.append("\n\n".join(current))
                current = []
            out.extend(_split_block(block, max_chars))
            continue
        if current and len("\n\n".join([*current, block])) > max_chars:
            out.append("\n\n".join(current))
            current = []
        current.append(block)
    if current:
        out.append("\n\n".join(current))
    return out


def _split_block(block: str, max_chars: int) -> list[str]:
    lines = block.split("\n")
    if lines[0].lstrip().startswith("|"):
        header_size = 2 if len(lines) > 1 and set(lines[1].replace(" ", "")) <= set("|-:") else 1
        return _group(lines[header_size:], max_chars, prefix=lines[:header_size])
    if len(lines) > 1:
        return _group(lines, max_chars)
    return _split_text(block, max_chars)


def _group(lines: list[str], max_chars: int, prefix: list[str] | None = None) -> list[str]:
    """Pack lines into groups of at most max_chars, each starting with `prefix`."""
    prefix = prefix or []
    budget = max_chars - (len("\n".join(prefix)) + 1 if prefix else 0)
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        parts = _split_text(line, budget) if len(line) > budget else [line]
        for part in parts:
            if current and len("\n".join([*current, part])) > budget:
                groups.append(current)
                current = []
            current.append(part)
    if current:
        groups.append(current)
    return ["\n".join(prefix + group) for group in groups]


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split running text at sentence ends, or at spaces for a huge sentence."""
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        for piece in _hard_wrap(sentence, max_chars):
            if current and len(current) + 1 + len(piece) > max_chars:
                out.append(current)
                current = ""
            current = f"{current} {piece}" if current else piece
    if current:
        out.append(current)
    return out


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars + 1)
        cut = cut if cut > 0 else max_chars
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return [*parts, text] if text else parts


def _breadcrumb(doc: Document, path: tuple[str, ...]) -> str:
    crumb = " › ".join([doc.title, *(_plain(p) for p in path)])
    year = doc.metadata.get("content_year")
    return f"{crumb} [{year}]" if year else crumb


def _plain(heading: str) -> str:
    return _EMPHASIS.sub("", _LINK.sub(r"\1", heading)).strip()


def _text(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in lines).strip()
