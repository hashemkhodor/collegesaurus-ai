# Collegesaurus AI — chatbot service

A FastAPI app that answers questions about Lebanese universities, majors and
scholarships from the pages on [collegesaurus.org](https://collegesaurus.org), with a
link to each source. It replaces the Streamlit app at the repo root, which keeps
running until cutover. The design and the reasons behind it are in
[`docs/chatbot-redesign.md`](../docs/chatbot-redesign.md).

```
collegesaurus site build ──► /chatbot/corpus.json (en, ar) + /chatbot/version.json
                                          │  polled every 60 s
                                          ▼
 chat page (web/) ◄──SSE── server.py ── agent.py (Gemini tool loop) ── tools.py
                              │                                           │
                              └─ logging_store.py → Supabase     store.py (SQLite + numpy)
                                                                          ▲
                              ingest.py: sources.py → chunking.py → Gemini embeddings
```

## Run it locally

Python 3.11 or newer. On macOS use a Homebrew, python.org or uv Python.

```bash
uv venv .venv --python 3.12
uv pip install -r chatbot/requirements-dev.txt
```

The chatbot reads the corpus the site publishes, so run a local build of the
collegesaurus site first. That needs the site's `chatbot-corpus` plugin. From
the collegesaurus checkout:

```bash
CHAT_URL=http://localhost:8000 ./scripts/sync-test.sh   # sync from Drive, then build
npm run serve                                            # http://localhost:3000
```

**Without a Gemini key (demo mode).** This runs the real app, with a word-overlap
embedder and a scripted model in place of Gemini. Answers are canned summaries of
the top search results.

```bash
.venv/bin/python -m scripts.demo_server --site http://localhost:3000
```

**With Gemini:**

```bash
export GEMINI_API_KEY=...  SITE_URL=http://localhost:3000
.venv/bin/python -m chatbot.ingest --query "AUB tuition"     # build data/index.db, try a search
.venv/bin/uvicorn chatbot.server:app --port 8000
```

Then open http://localhost:3000/universities/aub and click **Ask AI**, or open
http://localhost:8000 directly.

**Tests and lint:**

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check chatbot tests scripts && .venv/bin/ruff format --check chatbot tests scripts
```

## What the site publishes

The collegesaurus plugin `plugins/chatbot-corpus/` writes these files at build time.
`sources.py` reads them.

- **`/chatbot/version.json`**: `{drive_fingerprint, site_commit, built_at}`. It changes
  when the Drive content or the site code changes; the chatbot polls it every minute.
- **`/chatbot/corpus.json`** and **`/ar/chatbot/corpus.json`**:
  `{schema: 1, locale, content_sha, docs: [...]}`. Each doc has:
  - `type`: `university` or `scholarship`
  - `slug`
  - `title`: e.g. `AUB — American University of Beirut`
  - `url`: absolute, the page not a section
  - `content_locale`: the language the body is really in. An untranslated Arabic
    page carries the English body; the chatbot skips it and lets the English page
    answer Arabic questions.
  - `content_year`, `apply_url`
  - `body`: the page as clean markdown. `MajorsTable` becomes a markdown table and
    there is no JSX.

## Configuration

| Variable | Default | What it's for |
|---|---|---|
| `GEMINI_API_KEY` | required | Chat and embeddings |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash-lite` | `gemini-3.5-flash-lite` also works |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | 768-dimension vectors |
| `SITE_URL` | `https://collegesaurus.org` | Where the corpus comes from |
| `INDEX_PATH` | `data/index.db` | Index snapshot loaded at startup |
| `SUPABASE_URL`, `SUPABASE_SECRET_KEY` | empty (logging off) | Chat logs and feedback |
| `IP_HASH_SECRET` | empty (no IP hash) | Key for the HMAC of client IPs |
| `DAILY_TURN_LIMIT` | `1500` | Questions per day, all users together |
| `FRAME_ANCESTORS` | `https://collegesaurus.org http://localhost:3000` | Pages allowed to embed the chat |

Other rate limits are set in `config.py`: 10 turns per 30 s per browser session, and
30 per minute per IP.

## Deploy (Fly.io, one-time setup)

1. `fly apps create collegesaurus-ai`. The name and region are in `fly.toml`.
2. Set the runtime secrets:
   `fly secrets set GEMINI_API_KEY=… SUPABASE_URL=… SUPABASE_SECRET_KEY=… IP_HASH_SECRET=$(openssl rand -hex 32)`
3. Run [`supabase/chat_logs.sql`](../supabase/chat_logs.sql) in the Supabase SQL editor.
4. In GitHub (Settings → Secrets and variables → Actions) for this repo:
   - add the secrets `GEMINI_API_KEY` and `FLY_API_TOKEN` (from `fly tokens create deploy -a collegesaurus-ai`);
   - add the variable `FLY_DEPLOY_ENABLED` = `true`.
5. Run the **Deploy chatbot to Fly.io** workflow. It builds the index snapshot, bakes it
   into the image and deploys. Check `https://collegesaurus-ai.fly.dev/healthz`.
6. Add a free uptime monitor on `/healthz` with a keyword alert on `"stale":true`, and a
   Google Cloud budget alert for the Gemini project.
7. Optional custom domain: `fly certs add chat.collegesaurus.org` plus a CNAME record.

## Cutover from Streamlit

1. In the collegesaurus repo, merge the `chatbot-corpus` plugin and the ChatBubble
   change.
2. Set its Actions variable `CHAT_URL` to the Fly URL and re-run the site deploy.
   To roll back, set `CHAT_URL` back to the Streamlit URL.
3. About two weeks later:
   - remove the Streamlit files at the repo root, `chroma_db/` and the old pins;
   - move `chatbot/requirements.txt` to the root and point `.devcontainer` at uvicorn;
   - delete the Streamlit Cloud app;
   - in Supabase, enable RLS on `chat_logs` (see the end of the SQL file).

## Operations

- **Questions:** Supabase `chat_logs` has one row per turn, with outcome, cited sources
  and top retrieval score. `chat_feedback` holds the thumbs up/down. The `chat_gaps`
  view lists off-topic refusals, thumbs-down answers and weak matches.
- **Freshness:** `/healthz` shows the build id and `content_sha` being served, the last
  good poll and the last refresh error. `stale` turns true after 10 minutes of failed polls.
- **Refreshes:** the running app never swaps in an index that loses more than half its
  chunks or still contains JSX. The old one keeps serving instead.
