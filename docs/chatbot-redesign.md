# Collegesaurus AI: chatbot redesign proposal
_Proposal from argus task `collegesaurus-ai/rethink-chatbot`, researched 2026-09-24. Status: approved plan, not yet implemented. The Streamlit app keeps running until cutover._

## Context
collegesaurus-ai answers questions about Lebanese universities and scholarships from the
collegesaurus site's content. It runs on Streamlit Community Cloud, which puts the app to sleep
after 12 hours with no traffic; waking it takes a click and 30–60+ seconds.

The owner wants a redesign sized for a personal project:
- a simple chatbot backed by a vector database of the real site content;
- room for more data sources later, not built now;
- a clean chat UI;
- logging of questions and answers, to find content gaps;
- hosting that doesn't sleep.

## Recommendation
1. **Keep Python and the working Gemini tool loop. Replace Streamlit with a small FastAPI app.** It
   serves a lightweight chat page and a streaming `/api/chat` endpoint.
2. **Source: the site build publishes its own content.** The MDX that drive_sync generates is
   exported by the site build as clean markdown plus metadata, at
   `collegesaurus.org/chatbot/corpus.json` and `/ar/chatbot/corpus.json`. A small Docusaurus plugin
   does this. We don't scrape HTML or read Drive.
3. **The data stays current (owner requirement: "always up to date").**
   - **Chatbot follows the site within about 1 minute.** The app polls a small `/chatbot/version.json`
     every 60 s, bypassing the CDN cache. When the content hash changes it re-embeds only the chunks
     that changed and swaps in the new index within seconds. It needs no secrets or cross-repo token.
   - **Site follows Drive within about 10–30 minutes.** The site's deploy workflow moves from a
     nightly run to a cron every ~10 minutes. A quick Drive fingerprint check comes first, and the
     full sync, build and deploy only run when something in Drive changed.
   - **Fast restarts.** Each chatbot deploy bakes a snapshot index into the image, so a restart is
     up in seconds and then catches up at the next poll.
4. **Vector store: one SQLite file.** It holds the chunks and their 768-dimension `gemini-embedding-001`
   vectors, searched exactly in memory with numpy. It sits behind a `VectorStore` interface. This
   replaces the 41 MB `chroma_db/` in git and Chroma's heavy dependencies.
5. **Room for step two: a `Source` interface.** Adding a data source means one class plus one
   registry line. Nothing downstream changes.
6. **Logging: keep Supabase.** Keep `chat_logs`, add outcome, sources, score, page, model and index
   version, and store feedback in its own insert-only table. Replace the raw IP with an HMAC hash,
   use the new secret key, and make direct REST calls instead of supabase-py.
7. **Host: Fly.io.** One always-on 512 MB machine in Frankfurt, about $3.30/month. Deploys start the
   new machine before stopping the old one (bluegreen), triggered from GitHub Actions.
8. **Cutover means changing the site's `CHAT_URL` Actions variable.** Rolling back is the same change.

## 1. Current repo inventory (collegesaurus-ai @ ff2c834)
**Stack:**
- Python 3.11, with a Streamlit UI.
- `google-genai`: chat uses `gemini-2.5-flash-lite`; embeddings use `gemini-embedding-001` at 3072 dimensions.
- ChromaDB 0.5, embedded in the app.
- Supabase for chat logs.

The README (`README.md:5-40`) is out of date: it describes LlamaIndex, fastembed and Gemini 2.0.

| File | What it does | Verdict |
|---|---|---|
| `app.py` | Streamlit UI: EN/AR/Auto labels, right-to-left CSS and the language picker (`:24-162`). Suggested questions. A per-session limit of 10 messages per 30 s and a 1000-character cap (`:290-304`). X-Forwarded-For parsing, which never receives the client IP on Streamlit Cloud (`:170-214`). A `log_turn` call on each turn (`:323-334`). | **Rebuild** the UI. Port its strings, limits and suggestions. |
| `agent.py` | A hand-written Gemini function-calling loop (`ChatSession.send`, `:178-240`). A strict-scope system prompt with the `__out_of_scope__` sentinel (`:39-76`). `_generate_with_retry`, which retries once on 429/503, with an `__upstream_busy__` fallback (`:117-138`). `MAX_STEPS=6`. | **Reuse** it. Add streaming, generic tools and Gemini 3 compatibility. |
| `tools.py` | Four tools with hand-written `FunctionDeclaration`s (`:99-181`). `_search` returns the full chunk text plus a `Source:` URL. `TOP_K=15`, although the tool descriptions still say "up to 5" (`:139-140`). | **Reuse** the pattern. Merge the four tools into `search(query, type?)` and `list_pages(type)`. |
| `ingest.py` | Fetches pages from GitHub raw or `LOCAL_MDX_DIR` (`:35-80`). `strip_frontmatter` (`:91`); `split_markdown` splits on `##`, packs chunks to 1024 characters and overlaps them by 128 (`:104`); `embed_all` embeds in batches (`:178`). Drops and rebuilds one Chroma collection per content type (`:196`). | **Refactor**: fetch corpus → chunk → embed → store. |
| `logging_store.py` | A fire-and-forget Supabase insert into `chat_logs`: session_id, lang, question, answer, tool_calls, latency_ms, error, **raw ip**, user_agent | **Reuse** it, with extra fields and no raw IP. |
| `config.py` | Environment variables and constants | **Reuse** it and trim. |
| `chroma_db/` (41 MB, committed) | Built 2026-05-10. Universities: 370 chunks from 14 pages. Scholarships: 119 chunks from 9 pages. About 446k characters at 3072 dimensions, plus rows for 4 dead segment IDs from earlier builds. | **Retire** it. No index in git. |
| `requirements.txt`, `.streamlit/config.toml`, `.devcontainer/`, `static/logo.svg` | Dependencies, including opentelemetry/protobuf pins that exist only for Streamlit Cloud. The theme (`#00A651`). A Codespaces setup that runs Streamlit. The logo. | Keep the logo and colours. Retire the rest at cutover. |

**How it answers today:** agentic RAG. Gemini chooses a tool and writes the query, Chroma returns
the top 15 chunks, and Gemini answers citing the URLs. The design is already vector-store-shaped;
keeping one collection per content type is what doesn't extend.

**Broken or stale today:**
- **Ingest can't read current content.** Since collegesaurus `f71aef3` (2026-09-20), drive_sync writes
  `{universities,scholarships}_versioned_docs/version-<year>/<slug>.mdx`, with Arabic under
  `i18n/ar/...`. `ingest.py` expects `<dir>/universities/<slug>.mdx`, and a GitHub raw fetch finds only
  `_template.mdx`. The live bot is answering from the May index.
- **Citation links point to the old domain.** `SITE_BASE_URL` is still `hashemkhodor.github.io/collegesaurus`;
  the site now lives at `https://collegesaurus.org`.
- **Majors are indexed as raw JSX.** Pages contain `<MajorsTable rows={[{program: '…', …}]}/>`
  (256 blocks covering 610 programs, with `\uXXXX` escapes), and the indexer embeds that markup
  as-is.
- **Some chunks are poor.** Some are just a page title (22 characters), and the 128-character
  overlap cuts sentences in half.

## 2. Data source and vector store (step one)

### 2.1 Source: generated MDX, exported by the site build as clean markdown
- **Why the MDX.** It is drive_sync's canonical output and the exact text the site renders. Its
  frontmatter carries `title`, `content_year` and `apply_url`. Docusaurus supplies the permalinks
  for the newest year in each locale.
  - The MDX is gitignored and exists only inside the site's CI job, so the site build has to publish it.
  - `plugins/homepage-data/index.ts` already reads `loadedVersions[0].docs` (permalink, frontMatter,
    `source`) in `allContentLoaded`, so the new plugin can copy that pattern.
- **Cleanup happens in the site plugin**, next to the format it cleans.
  - `<MajorsTable rows={[…]}/>` becomes a markdown table. Rows are parsed as JSON5 and keep the
    program, degree, department, credits, years and language columns, with `source` as a link.
  - `<div className="alert-…">` keeps its text only.
  - A `:::warning[Not yet updated …]` block becomes a one-line note.
  - `\<`, `\{` and `\}` are unescaped, and `> TODO` blockquotes written by editors are dropped.
  - The build warns if an unknown JSX component remains, as `homepage-data` already does at `:98-106`.
    This keeps the chatbot unaware of the site's components: the contract is markdown plus metadata.
- **Arabic fallbacks.** `info.ar.docx` is optional, and an Arabic page without a translation is
  built from the English file. The plugin marks each doc's real `content_locale` from its source
  path (`@site/i18n/ar/…` or not). The French branch adds a similar field.
- **Why not the published HTML.**
  - Since 2026-09-23 `src/remark/remarkGuidebook.mjs` restructures the pages: a program explorer,
    restyled tables and derived facts.
  - The UI is also changing (French, the calendar widget), so a scraper would chase markup and have
    to strip navigation.
  - The lunr `search-index.json` flattens tables.
- **Why not Drive.**
  - The chatbot would duplicate drive_sync's docx/xlsx parsing.
  - It would need the Drive service account in a second repo.
  - drive_sync can't run offline from `.drive-cache`.

### 2.2 Flow and freshness (target: a Drive edit is live on the site and in the chatbot in ~10–30 minutes)
```
site deploy.yml, every ~10 min (plus every push or manual run):
  check-drive job (~30 s): Drive fingerprint = hash of (file id, md5Checksum, modifiedTime) for every file under the root
     compared with the drive_fingerprint in the live /chatbot/version.json. Same → stop (scheduled runs only).
     Changed, and last edit more than 3 min ago → continue.
  build: drive_sync ─► MDX ─► Docusaurus build ─► GitHub Pages (collegesaurus.org)
         └─[new] chatbot-corpus plugin ─► /chatbot/corpus.json, /ar/chatbot/corpus.json (each with its own content_sha),
                                          /chatbot/version.json {drive_fingerprint, site_commit, built_at}
chatbot app (always on):
  boot: load the snapshot baked into the image (instant) ─► poll right away
  every 60 s: GET version.json?t=<now> ─► drive_fingerprint or site_commit changed? ─► GET both corpora (cache-busted)
              ─► re-index only the locales whose content_sha changed: re-chunk, embed only new chunks
                 (reuse vectors by text hash) ─► index guard ─► swap in memory (seconds)
chatbot deploy workflow (push to main): fetch corpus ─► build index.db snapshot ─► bake into image ─► flyctl deploy
```
- **How long a Drive edit takes to show up.**
  - Up to about 10 minutes until the next cron run; GitHub cron can run late.
  - About 3 minutes of quiet after the last edit, so half-finished edits aren't published.
  - About 5 minutes for drive_sync, the build and the Pages deploy.
  - About 1 minute for the chatbot poll and re-embedding.
  - Typical total: about 10–20 minutes; worst case about 30.
  - A push or manual run skips the cron wait.
- **Why a fingerprint check.** Most runs find no change and finish in about 30 seconds.
  - It needs a small `python -m drive_sync --fingerprint` mode that walks the Drive tree without downloading.
  - It reuses the existing service-account secret, so there's no new secret.
  - Actions minutes are free for public repos.
- **Why the chatbot polls instead of the site pushing.**
  - Polling needs no shared secret and no cross-repo token.
  - It heals itself after restarts or missed deploys.
  - The version file is about 100 bytes, so 1,440 polls a day cost almost nothing.
  - The `?t=` parameter skips GitHub Pages' 10-minute CDN cache.
  - collegesaurus-ai is dormant (last pushed 2026-05-10), and GitHub turns off scheduled workflows
    after 60 days without activity, so a cron there would stop.
- **What the baked snapshot buys.**
  - Restarts and crashes come back in seconds, without needing the site or Gemini.
  - It seeds vector reuse, so a typical content change re-embeds only a few chunks.
  - A full re-embed is about 0.3M tokens, roughly $0.05.
- **`content_sha` hashes page content only,** so a rebuild with no edits doesn't trigger a re-index.
- **Watch for staleness.** The site's own schedule also stops after 60 days with no commits
  (GitHub emails a warning first). `/healthz` reports how old the index is, and step 8 adds an
  uptime monitor.
- **Index guard.** A refreshed index is refused if:
  - the doc or chunk count drops by more than half;
  - any `<Capitalized` JSX tag is left in the text;
  - the embedding call fails.
  The old index keeps serving, and `/healthz` reports the error.

### 2.3 What gets indexed, and chunking
- **Scope.** Only the newest academic year: the version served at `/universities/<slug>`.
  - That is 14 universities and 9 scholarships.
  - It includes the English pages, plus each Arabic page that is really in Arabic.
  - `_template` is skipped.
- **Keep `content_year`.** 13 of the 14 university pages are 2025-2026 content carried forward, so
  answers should say which year a figure comes from.
- **Chunking.**
  - Split on H2, then H3 (faculties), then by table rows (repeating the header row) or by paragraphs.
  - Aim for about 1.2k characters and cap at about 2.5k. Merge the title-only stub into the next
    chunk, and drop the character overlap.
  - Start every chunk with a breadcrumb, e.g. `AUB — American University of Beirut › Faculty › MSFEA (2026-2027)`.
- **Metadata:** `source, type, slug, locale, title, url, section, content_year`.
  - `url` is the page, not the section anchor, because anchors change every year and in every language.
  - IDs look like `collegesaurus:en:university/aub:7`.
- **Search language.** Arabic questions search Arabic chunks, plus the English chunks of pages that
  have no Arabic translation. All other questions search the English chunks.

### 2.4 Embeddings and vector store
- **Embeddings: `gemini-embedding-001`.**
  - It is stable until 2028-05-14, supports Arabic, and takes a task type for documents and queries.
  - Use its 768-dimension output, L2-normalized by us (required below 3072).
  - To switch to `gemini-embedding-2` later, change the config and redeploy.
- **Vector store: `SqliteNumpyStore` behind a `VectorStore` interface.**
  - The index is one SQLite file written with the standard library: `chunks` rows with the text,
    metadata and vector as a BLOB, plus a `meta` table (content_sha, model, dim, built_at).
  - At load time the vectors become an N×768 float32 matrix, and search is an exact cosine match
    with boolean metadata masks. About 2k vectors is roughly 6 MB, and a query takes under 1 ms.
  - The file opens in any SQLite browser if you want to see what was indexed.
  - A swap is just a Python object reassignment.
- **Why no vector-database engine.**
  - At this size an engine adds nothing.
  - sqlite-vec is pre-1.0, needs SQLite extension loading (which the macOS system Python lacks),
    and has an open report of cross-platform corruption.
  - chromadb 1.x pulls in onnxruntime, OpenTelemetry, grpc and kubernetes (about 100 MB of wheels,
    the source of today's protobuf pins), and moving 0.5 data to 1.x is unreliable.
  - If a step-two source makes the corpus 100× bigger, switch the `VectorStore` implementation to
    sqlite-vec, pgvector or Chroma. Nothing else changes.

## 3. Extensibility for step two (describe only; not built now)
```python
@dataclass(frozen=True)
class Document:            # every source yields clean markdown + metadata
    source: str; doc_id: str; type: str; locale: str
    title: str; url: str | None; body: str; metadata: dict   # e.g. content_year

class Source(Protocol):
    name: str
    types: dict[str, str]          # type → one-line description (feeds tool enum + scope prompt)
    def fingerprint(self) -> str   # cheap change check (content_sha / ETag)
    def load(self) -> list[Document]
```
- **Registry.** Sources are listed in config: `SOURCES = [CollegesaurusCorpus(url_en, url_ar)]`.
  - Each source refreshes on its own, via `store.replace_source(name, chunks)`, so one source failing
    or changing leaves the others alone.
  - Cleaning a publisher's own format is the publisher's job, as with the site plugin. For sources we
    don't control, the cleanup lives inside that `Source` class.
- **Everything downstream ignores the source:**
  - The chunker, embedder and store are shared.
  - The tools are generic: `search(query, type?)` and `list_pages(type)`. Their `type` enum, their
    descriptions and the scope sentence in the system prompt are all built from the registered sources.
  - Logs record the source of every cited chunk.
- **Adding a source is one class plus one registry line.** The server, UI, store schema and logging
  don't change.
  - Candidates: `src/data/homepage/deadlines.ts`, the derived facts in `design/university-page`,
    and the share-experience GitHub issues. (`stories/` was added on 2026-09-24 at the owner's
    request, as a `story` type in the site's corpus rather than a separate source.)
  - A much larger source, such as crawled university sites, would only swap the `VectorStore`.

## 4. How answers are produced (mostly reused)
- **Kept:** `agent.py`'s tool loop, the prompt rules, the 429/503 retry and both sentinels.
- **Changes:**
  - **Tools:** the two generic tools; `SITE_BASE_URL` becomes `https://collegesaurus.org`.
  - **Streaming** uses `generate_content_stream`. Function calls arrive whole in one chunk. Only the
    final answer streams to the user.
  - **429/503 retry** happens only before the first text is sent. After that, an error ends the turn
    with a `busy` or `error` event.
  - **Text before a tool call.** If the model writes text and then calls a tool, the server sends a
    `discard` event and the UI drops that partial text.
  - **Out-of-scope sentinel** is matched after trimming whitespace, backticks and quotes, using
    `startswith` within the first ~40 characters. It is buffered until then, so it's never shown.
  - **History comes from the browser but is checked.** Only `user` and `assistant` roles are
    accepted, as plain text. Limits: 10 turns, 1,000 characters per user turn, 4,000 per assistant
    turn, and a 32 KB body. Anything else gets a 400. Forged history can't change the system prompt,
    and rate limits bound what it can cost.
  - **Page context (optional):** "the user is viewing the AUB page", taken from the iframe's `page=` parameter.
- **Model: keep `gemini-2.5-flash-lite`** ($0.10 / $0.40 per million tokens).
  - It has no shutdown date, but since 2026-09-18 it is only served to existing projects like this one.
  - Write the loop so it also works with Gemini 3: keep the model's full `Content` (it carries thought
    signatures), echo the function-call `id`, and don't send `temperature` to 3.x models. Moving to
    `gemini-3.5-flash-lite` ($0.30 / $2.50) is then a config switch.
  - Pin `google-genai>=2.25,<3`.
  - Cost is roughly $1 per 1,000 questions on 2.5, and about $4 on 3.5.

## 5. UI/UX recommendation
**One chat page, served by the app.** It is designed first for the site's existing 400×600 bubble
panel, which fills the screen on phones (`src/components/ChatBubble/index.tsx`,
`styles.module.css:155-163`), and it also works as a standalone page.
- **Keep the existing iframe bubble.** Cutover is just changing `CHAT_URL`.
- **Don't build a native React widget inside Docusaurus now.** It would tie site builds to the
  chatbot API and need CORS. Revisit only if you want features inside the pages themselves.
```
┌ Collegesaurus AI ─────────────── × ┐ ← site's existing panel header
│ EN | عربي                ↺ New chat │ ← slim toolbar (?embed=true hides the page header)
│ 🦕 Hi! Ask me about Lebanese         │
│ universities, majors & scholarships. │
│ [AUB tuition?] [Scholarships abroad] │ ← 4 localized suggestion chips (from app.py)
│               ┌───────────────────┐  │
│               │ USJ medicine reqs?│  │
│               └───────────────────┘  │
│ ┌──────────────────────────────────┐ │ ← streamed markdown; tables scroll sideways;
│ │ **Medicine at USJ** requires …   │ │   "Searching universities…" while tools run
│ └──────────────────────────────────┘ │
│ Sources: [USJ ↗] [LAU ↗]      👍 👎  │ ← cited pages + feedback
│ [Ask about a university, major… ] ➤ │ ← Enter sends · char counter near limit
│ Questions are logged anonymously to  │
│ improve the site. Please don't       │
│ include personal details.            │
└──────────────────────────────────────┘
```
- **Site parameters.** The site passes none today. A small ChatBubble change is part of the plan:
  - `lang` (en, ar, fr) from `i18n.currentLocale`.
  - `page`, captured when the panel opens. The bubble stays mounted across page navigation, so a
    reactive URL would reload the iframe.
  - An optional `theme`.
  - Coordinate this change with the French branch, which also edits ChatBubble.
  - **Without the parameters:** the English UI, answers in whatever language the user writes, and no
    page context.
- **Language behaviour.** A toggle switches between English and Arabic; Arabic flips the layout to
  right-to-left, and each bubble uses `dir="auto"`. Replies follow the language the user writes in.
- **Keeping the conversation.** The bubble unmounts the iframe when it closes, so the conversation
  is kept in `sessionStorage`. Every access is wrapped in try/catch.
- **Links.** Site links open in the parent tab; external links open in a new tab.
- **Errors.** Friendly messages for busy, rate limited and too long.
- **Tech:**
  - Plain `index.html`, `chat.js` and `chat.css` with no build step.
  - Vendored `marked` and `DOMPurify` render the markdown safely.
  - Theme colour `#00A651` and the existing logo.

## 6. Question logging
- **Where.** Supabase `chat_logs`, through a ported `logging_store.py` that calls Supabase's REST
  API directly with `httpx` (already a dependency), instead of the heavy supabase-py SDK.
  - The row is written in the stream's `finally` block, so a turn is logged even if the user leaves.
    That turn gets the outcome `aborted`.
  - Logging failures never affect the chat.
- **One row per turn.** It keeps `session_id, lang, question, answer, tool_calls, latency_ms, error`
  and adds:
  - `turn_id` (uuid), `page`, `model`, `index_version`, `ip_hmac`
  - `outcome`: answered, out_of_scope, busy, error or aborted
  - `sources`: the cited URLs
  - `top_score`: the best retrieval score, a signal of content gaps
- **Feedback.** Thumbs up or down go into a separate insert-only `chat_feedback` table
  (`turn_id, value, created_at`), so feedback never races the log insert.
- **Reviewing.** A saved "content gaps" query or view lists out-of-scope questions, thumbs-down
  answers and low `top_score` turns, newest first. The DDL is versioned in
  `supabase/chat_logs.sql`, which doesn't exist today.
  - Supabase projects stop auto-granting new `public` tables from 2026-10-30, so the migration needs
    an explicit `GRANT INSERT … TO service_role`.
- **Supabase changes that are needed anyway:**
  - **New key.** Use the `sb_secret_…` server key, and confirm which header form the REST call needs.
    The legacy anon and service_role keys are deprecated by the end of 2026, and a paused project
    that gets restored won't get them back.
  - **Keepalive.** The always-on app runs a small read about every 4 hours. Otherwise a quiet week
    pauses the free project, and while paused it rejects writes with error 540.
- **Privacy, in one line.** Store the question, the answer and an HMAC-SHA256 of the IP under a
  secret key, for abuse correlation.
  - No raw IP, no user agent and no accounts.
  - The UI notes that questions are logged anonymously.
  - On the paid Gemini tier, prompts aren't used to improve Google's products.

## 7. Hosting comparison (researched 2026-09-24)
| Option | Always-on cost | Sleep / cold start | Setup & upkeep | Notes |
|---|---|---|---|---|
| **Fly.io (recommended)** | ~$3.30/month (shared-cpu-1x, 512 MB) | None with `auto_stop_machines="off"` | Low: Dockerfile, `fly.toml`, official GitHub Action | Card required; single machine; EU region `fra` |
| Railway Hobby | $5/month flat (about $3 of usage, covered) | None; sleep is opt-in | Very low: git push | 8-hour platform outage in May 2026 |
| Render Starter | $7/month (0.5 CPU, 512 MB) | None on paid plans. The free tier sleeps after 15 min and takes about 1 min to wake. | Very low | Deploy hooks; tight on RAM |
| Cloud Run | $0 when it scales to zero, or ~$9.70/month kept warm | 1–3 s cold start unless kept warm | Medium (GCP IAM) | Same Google Cloud project as Gemini |
| Cloudflare Workers | $0–5 | Almost none (Python about 1 s) | High: rewrite for Workers, Vectorize and D1 | Free plan: 128 MB memory and 10 ms CPU |

**Rejected:**
- Streamlit Cloud: the status quo.
- Hugging Face Spaces: about $31/month to avoid sleeping.
- Koyeb: new users must pay at least $29/month.
- Vercel Hobby: cold starts, and non-commercial use only.
- Oracle Always Free: idle VMs get reclaimed.
- Hetzner: about €6/month, but you manage the server yourself.

**Why Fly.io:**
- It's the cheapest option that is truly always on.
- It runs a plain Docker image, so moving to Railway or Render is easy.
- Bluegreen deploys and the baked index mean deploys and restarts cause almost no downtime.
- Frankfurt is close to Lebanon.
- Railway is just as good if you prefer a dashboard, for $1.70 more per month.

## 8. Step-one implementation plan (for follow-up tasks)
The Streamlit app on `main` must keep working until cutover. So:
- New code goes in a new `chatbot/` package.
- The root `app.py`, `agent.py`, `tools.py`, `ingest.py`, `config.py`, `logging_store.py`,
  `requirements.txt` and `chroma_db/` stay untouched.
- New dependencies go in `chatbot/requirements.txt`. Don't add a root `pyproject.toml` or
  `uv.lock`, because Streamlit Cloud would pick those up.
- The new dependencies are fastapi, uvicorn, `google-genai<3`, numpy, httpx and python-dotenv, plus
  pytest and ruff for development.

1. **Site repo (collegesaurus), a separate task: corpus export and a Drive sync every 10 minutes.**
   - Add `plugins/chatbot-corpus/index.ts`, modelled on `plugins/homepage-data/index.ts`.
     - In `allContentLoaded`, collect the newest docs of `universities` and `scholarships` and keep them in a closure.
     - In `postBuild`, which runs once per locale, read each `source` file, clean it (section 2.1),
       and write `<outDir>/chatbot/corpus.json`:
       `{schema:1, locale, content_sha, docs:[{type, slug, title, url: siteConfig.url+permalink, content_locale, content_year, apply_url, body}]}`.
       Write it in `postBuild`, not earlier: the output directory is wiped during each locale's
       build (`buildLocale.js:65`).
     - The default locale also writes `/chatbot/version.json`:
       `{drive_fingerprint: $DRIVE_FINGERPRINT, site_commit: $GITHUB_SHA, built_at}`.
   - Register the plugin in `docusaurus.config.ts`.
   - Add a `python -m drive_sync --fingerprint` mode.
     - It reuses the existing tree walk in `fetch.py` without downloading anything, and prints the
       sha256 of the sorted `(file id, md5Checksum, modifiedTime)` list plus the newest `modifiedTime`.
     - Unit-test it with a mocked listing.
   - Edit `.github/workflows/deploy.yml`:
     - Change the cron from `'0 4 * * *'` to `'3-59/10 * * * *'`.
     - Add a first job, `check-drive`, that computes the fingerprint and compares it with the live
       `https://collegesaurus.org/chatbot/version.json?t=<now>`.
       - It outputs `changed=true` when they differ, the newest edit is more than 3 minutes old, and
         the run was triggered by the schedule.
       - Push and manual runs always build.
     - `build` gets `if: github.event_name != 'schedule' || needs.check-drive.outputs.changed == 'true'`
       and passes `DRIVE_FINGERPRINT` along.
     - Keep the existing `concurrency: pages` so runs queue instead of overlapping.
   - Add ChatBubble parameters `lang`, `page` (captured on open) and `theme`.
   - Replace the hard-coded Streamlit fallback URLs (`docusaurus.config.ts:133-134`,
     `ChatBubble/index.tsx:26-28`) and the out-of-date comments (`deploy.yml:39-56`).
   - Coordinate with the French branch.
2. **Ingestion.**
   - `chatbot/sources.py`: `Document`, `Source` and `CollegesaurusCorpus`.
   - `chatbot/store.py`: `VectorStore` and `SqliteNumpyStore`.
   - `chatbot/ingest.py`: the chunker, the Gemini embedder with text-hash reuse, refresh, the guard and the swap.
   - A command line: `python -m chatbot.ingest --corpus <url|file> --out data/index.db --query "AUB tuition"`.
3. **Agent.** Port `chatbot/tools.py` and `chatbot/agent.py`: generic tools, language filtering,
   streaming with `discard`, retry before the first text only, robust sentinel matching, Gemini 3
   compatibility, and extraction of cited sources.
4. **Server: `chatbot/server.py` (FastAPI).**
   - Routes:
     - `GET /` serves the static page.
     - `POST /api/chat` streams server-sent events: `status`, `delta`, `discard`, `sources`, `done`.
     - `POST /api/feedback` records a rating.
     - `GET /healthz` returns 200 once an index is loaded. Its body shows the build id, the
       `content_sha` for each locale, when the last poll succeeded, the last refresh error, and
       `stale: true` if polling has failed for more than 10 minutes.
   - Load the snapshot at startup, then run two background tasks: the 60-second `version.json`
     poll and refresh (section 2.2; at most one refresh at a time), and the Supabase keepalive.
   - History validation as in section 4.
   - Limits, with the client IP taken from `Fly-Client-IP`:
     - 10 turns per 30 s per session;
     - 30 turns per minute per IP;
     - 1,500 turns per day overall;
     - 1,000 characters per input.
     There's no strict daily cap per IP, because carrier NAT and school networks share IPs.
   - Header: `CSP frame-ancestors 'self' https://collegesaurus.org http://localhost:3000`.
5. **UI.** `chatbot/web/{index.html,chat.js,chat.css,vendor/}`, as in section 5.
6. **Logging.** Port `chatbot/logging_store.py` (httpx, secret key, HMAC IP hash) and add a
   `supabase/chat_logs.sql` migration: the new columns, the `chat_feedback` table with grants, and
   the gaps view.
7. **Shipping.**
   - `Dockerfile` (python:3.12-slim, non-root user, `uvicorn --proxy-headers`, copies `data/index.db`) and `.dockerignore`.
   - `fly.toml`:
     - `primary_region="fra"`, `auto_stop_machines="off"`, `min_machines_running=1`, 512 MB;
     - a `/healthz` check;
     - `[deploy] strategy="bluegreen"`.
   - `.github/workflows/deploy.yml`:
     - Runs on push to main or a manual run.
     - Builds `data/index.db` from the live corpus (needs a `GEMINI_API_KEY` secret), then runs
       `flyctl deploy --remote-only`.
     - Fails without deploying if the corpus fetch or the guard fails.
   - `ci.yml`: ruff and pytest on every pull request.
   - Runtime secrets are set with `fly secrets set`.
8. **Operations.**
   - A free uptime monitor on `/healthz` with email alerts. It also checks the response for the
     keyword `"stale":true`, so a sync that stops quietly still raises an alert.
   - A Google Cloud billing budget alert for the Gemini project.
   - An optional custom domain, `chat.collegesaurus.org`.
9. **Deploy and soak.** Run the checks in section 10 against `collegesaurus-ai.fly.dev`, then leave
   it idle overnight.
10. **Cutover.** Set collegesaurus `CHAT_URL` to the new URL and re-run the site deploy. To roll
    back, restore the Streamlit URL.
11. **Cleanup, about 2 weeks later.**
    - Remove the Streamlit files, `chroma_db/`, `.streamlit/` and the pins; move the dependencies to the root.
    - Rewrite the README and point `.devcontainer` at uvicorn.
    - The owner deletes the Streamlit Cloud app.
    - Normal commits only; no history rewrite.

**Out of scope (step two and later):**
- Any second data source: deadlines, derived facts, experiences, external sites. (Stories were
  added on 2026-09-24, at the owner's request.)
- A native React widget.
- Accounts or an admin UI.
- Hybrid keyword + vector search, unless the evaluation shows misses.
- Answer caching, dashboards, and running in more than one region.

## 9. Open questions for the owner (my default in brackets)
**Already decided (owner, 2026-09-24):** data must stay current.
- The chatbot follows the live site within about 1 minute.
- The site checks Drive every ~10 minutes, so a Drive edit reaches the site and chatbot in about 10–30 minutes (section 2.2).

1. **Gemini terms and a teen audience.** The Gemini API terms (effective 2026-03-23) say it must not
   be used in a service "directed towards or … likely to be accessed by individuals under the age of
   18". The bot's own prompt targets high-school students (`agent.py:40-41`), so this applies to the
   current app too. Do you accept that, or should we evaluate another provider? The model and the
   embeddings are isolated in `agent.py` and `ingest.py`. [Decide before launch.]
2. **Hosting and budget.** Fly.io at about $3–4/month (a card is required), Railway at $5, or Render at $7? [Fly.io]
3. **Model.** Stay on `gemini-2.5-flash-lite`, or move to `gemini-3.5-flash-lite` now (3–6× the price)?
   [Stay, and compare later on the evaluation set.]
4. **Moderation and abuse.** Logs are private, so they need no moderation. Are rate limits, the
   daily cap and scope refusals enough, or do you want word filters or blocklists? [Enough.]
5. **Logging.** Drop the raw IP and user agent in favour of an HMAC IP hash? Should old logs be
   deleted after some time? [Yes; no deletion for now.]
6. **Supabase.** Is the project on the free plan, and in which region (Fly should run near it)? Is
   it OK to switch to the new secret key? [Yes, Frankfurt.]
7. **Access.** Public and anonymous, protected by rate limits and a daily cap? [Yes, 1,500 turns a day.]
8. **Site changes.** Is it OK to add the corpus plugin and the ChatBubble parameters to the
   collegesaurus repo, as a separate task coordinated with the French branch?
9. **Content scope.** Only the newest year? Index Arabic too? French UI once the site ships French?
   [Yes, yes, yes.]
10. **Domain and Streamlit.** Use `chat.collegesaurus.org` or `*.fly.dev`? How long should Streamlit
    stay up after cutover? [Custom domain; about 2 weeks.]

## 10. Verification (for the implementation)
- **Unit tests (pytest).**
  - Site plugin: MajorsTable becomes a table, `\u00e9`-style escapes are decoded, alerts and banners are handled,
    `> TODO` is dropped, the unknown-component warning fires, and `content_locale` is set for Arabic
    fallbacks. Use the drive_sync test fixtures.
  - Chunker: breadcrumbs are present and there are no stub chunks.
  - Store: type and language masks, Arabic fallback, vector reuse, and the guard refusing a collapsed
    or JSX-laden index.
  - Refresh loop, with a fake HTTP layer:
    - A new build id fetches both corpora.
    - A locale whose `content_sha` didn't change is left alone and nothing is re-embedded.
    - A changed locale re-embeds only its new chunks.
    - A failed fetch or failed guard keeps the old index and eventually sets `stale`.
  - Site: `drive_sync --fingerprint` gives the same hash for the same listing and a different one
    when any `modifiedTime` changes. `check-drive` skips when the newest edit is under 3 minutes old.
  - Agent, with a fake Gemini client: a tool call then streamed text, `discard` after a preamble,
    sentinel buffering, and no retry after the first text.
  - Server: event order, history validation (400), rate limits (429), `/healthz` before and after
    load, and the CSP header.
- **Retrieval evaluation.** About 20 English and Arabic questions, each with the expected page
  slugs, reporting hit@5. Run it before cutover and after any change to chunking or the model.
- **Local end-to-end.** Run uvicorn against the live corpus. Try:
  - a list question, a majors table, tuition, and a follow-up question;
  - an Arabic question, checking the right-to-left layout;
  - an off-topic question, which should be refused;
  - thumbs up or down, and check the `chat_feedback` row appears.
  - Then run the local site with `CHAT_URL=http://localhost:8000` and check the 400×600 panel, the
    full-screen mobile panel, and closing and reopening.
- **Production.**
  - `/healthz` shows the snapshot and then the refreshed `content_sha`.
  - After 12+ hours idle, the first message answers instantly.
  - After `fly machine restart` the app is healthy again within seconds.
  - **Freshness, end to end.** Edit a test value in a Drive doc, such as a scholarship deadline.
    - Within about 10–30 minutes the `check-drive` job reports `changed` and the site shows the new value.
    - Within about 1 more minute, the `/healthz` build id and `content_sha` update.
    - The chatbot then answers with the new value, citing the page.
    - A scheduled run with no Drive change stops after `check-drive`, in about 30 seconds.

## 11. Next steps
Step one is implemented in follow-up tasks. This document is only the proposal.
1. **collegesaurus (site repo):**
   - the `chatbot-corpus` plugin (`corpus.json` + `version.json`);
   - `drive_sync --fingerprint` and the 10-minute `check-drive` cron in `deploy.yml`;
   - the ChatBubble `lang`, `page` and `theme` parameters, plus the Streamlit fallback-URL and comment cleanup.
   Coordinate this with the French-language branch.
2. **collegesaurus-ai (this repo):** the `chatbot/` package, including:
   - ingestion and the SQLite + numpy store;
   - the streaming agent and the FastAPI server;
   - the chat UI;
   - Supabase logging and its migration;
   - the Dockerfile, `fly.toml` and GitHub workflows.
   Then: the Fly.io deploy, the soak, the `CHAT_URL` cutover and the Streamlit cleanup.
3. **Before launch:** answer the open questions in section 9, especially the Gemini terms and the teen audience.
