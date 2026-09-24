"""Where the chatbot's knowledge comes from.

A Source yields Documents: clean markdown plus metadata. Everything after this
module (chunking, embedding, the store, the tools) works on Documents and never
needs to know which site or format they came from, so adding a data source
later means one new Source class plus one entry in config.SOURCES.

The collegesaurus site publishes its pages for us at build time: the site's
chatbot-corpus plugin writes one corpus.json per locale (already cleaned to
markdown) and a small version.json that changes on every build. The contract
is documented in chatbot/README.md.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

SCHEMA_VERSION = 1
REQUIRED_FIELDS = ("type", "slug", "title", "url", "body")


class CorpusError(Exception):
    """The published corpus is missing, unreachable or malformed."""


@dataclass(frozen=True)
class Document:
    source: str
    doc_id: str  # stable within its source, e.g. "university/aub"
    type: str  # "university", "scholarship", ...
    locale: str  # language the body is written in
    serves: tuple[str, ...]  # search locales this document answers for
    title: str
    url: str
    body: str  # clean markdown
    metadata: dict[str, str] = field(default_factory=dict, hash=False)


class Source(Protocol):
    name: str
    # type -> one-line description; feeds the search tool's enum and the prompt.
    types: dict[str, str]

    def fingerprint(self) -> str:
        """Cheap check that changes whenever load() would return new content."""
        ...

    def load(self) -> list[Document]: ...


class CollegesaurusCorpus:
    """Pages published by the collegesaurus site build (see module docstring).

    `corpora` maps locale -> URL or local path of that locale's corpus.json.
    With `version` (the URL of version.json) the fingerprint costs one tiny
    request; without it, the fingerprint hashes the corpus files themselves.
    """

    name = "collegesaurus"
    types = {
        "university": (
            "Lebanese university pages: faculties and majors, tuition and fees, "
            "admission requirements, application steps and contacts."
        ),
        "scholarship": (
            "Scholarship pages: eligibility, benefits, application windows, "
            "supported universities and how to apply."
        ),
    }

    def __init__(
        self,
        corpora: dict[str, str],
        version: str | None = None,
        *,
        client: httpx.Client | None = None,
        default_locale: str = "en",
    ):
        self._corpora = dict(corpora)
        self._version = version
        self._client = client or httpx.Client(timeout=20.0, follow_redirects=True)
        self._default_locale = default_locale

    def fingerprint(self) -> str:
        if self._version:
            data = json.loads(self._read(self._version))
            return f"{data.get('drive_fingerprint', '')}|{data.get('site_commit', '')}"
        digest = hashlib.sha256()
        for locale in sorted(self._corpora):
            digest.update(locale.encode())
            digest.update(self._read(self._corpora[locale]))
        return digest.hexdigest()[:16]

    def load(self) -> list[Document]:
        corpora = {
            locale: self._read_corpus(locale, location)
            for locale, location in self._corpora.items()
        }
        # (type, slug) pages that each locale really has in its own language.
        # An untranslated page is a copy of the English one; we skip the copy
        # and let the English document answer that locale's searches instead.
        translated = {
            locale: {(e["type"], e["slug"]) for e in corpus["docs"] if _is_own(e, locale)}
            for locale, corpus in corpora.items()
        }
        docs: list[Document] = []
        for locale, corpus in corpora.items():
            for entry in corpus["docs"]:
                if not _is_own(entry, locale):
                    continue
                key = (entry["type"], entry["slug"])
                serves = (locale,)
                if locale == self._default_locale:
                    serves += tuple(
                        other
                        for other in corpora
                        if other != locale and key not in translated[other]
                    )
                docs.append(
                    Document(
                        source=self.name,
                        doc_id=f"{entry['type']}/{entry['slug']}",
                        type=entry["type"],
                        locale=locale,
                        serves=serves,
                        title=entry["title"],
                        url=entry["url"],
                        body=entry["body"],
                        metadata={
                            k: entry[k] for k in ("content_year", "apply_url") if entry.get(k)
                        },
                    )
                )
        return docs

    def _read_corpus(self, locale: str, location: str) -> dict:
        try:
            corpus = json.loads(self._read(location))
        except ValueError as exc:
            raise CorpusError(f"{location}: not valid JSON ({exc})") from exc
        if corpus.get("schema") != SCHEMA_VERSION:
            raise CorpusError(f"{location}: unsupported schema {corpus.get('schema')!r}")
        if corpus.get("locale") != locale:
            raise CorpusError(
                f"{location}: expected locale {locale!r}, got {corpus.get('locale')!r}"
            )
        docs = corpus.get("docs")
        if not isinstance(docs, list):
            raise CorpusError(f"{location}: 'docs' must be a list")
        for entry in docs:
            missing = [k for k in REQUIRED_FIELDS if not isinstance(entry.get(k), str)]
            if missing:
                raise CorpusError(f"{location}: doc {entry.get('slug')!r} is missing {missing}")
        return corpus

    def _read(self, location: str) -> bytes:
        if not location.startswith(("http://", "https://")):
            try:
                return Path(location).read_bytes()
            except OSError as exc:
                raise CorpusError(f"{location}: {exc}") from exc
        # GitHub Pages caches for 10 minutes; a unique query string goes to origin.
        try:
            response = self._client.get(location, params={"t": str(time.time_ns())})
        except httpx.HTTPError as exc:
            raise CorpusError(f"{location}: {exc}") from exc
        if response.status_code != 200:
            raise CorpusError(f"{location}: HTTP {response.status_code}")
        return response.content


def _is_own(entry: dict, locale: str) -> bool:
    """True when the entry has content written in `locale` (not a fallback copy)."""
    return bool(entry["body"].strip()) and entry.get("content_locale", locale) == locale
