# Collegesaurus AI

Agentic-RAG chatbot for [Collegesaurus](https://hashemkhodor.github.io/collegesaurus) — answers questions about Lebanese universities and external scholarships using the site's own MDX pages as the knowledge base.

Stack:
- **Streamlit** for the chat UI
- **LlamaIndex** `ReActAgent` for the agent loop
- **ChromaDB** (embedded) for the vector store
- **Gemini 2.0 Flash** for generation, **`text-embedding-004`** for embeddings — both free via Google AI Studio
- **No torch, no HuggingFace downloads** — the whole thing installs in ~30 seconds

## Architecture

```
┌──────────────┐     ┌────────────────────┐     ┌─────────────────┐
│ Streamlit UI │────▶│  ReActAgent (LLM)  │────▶│ Gemini 2.0 API  │
└──────────────┘     └─────────┬──────────┘     └─────────────────┘
                               │ calls
                               ▼
              ┌──────────────────────────────────┐
              │ tools: search_universities,      │
              │        search_scholarships,      │
              │        list_universities,        │
              │        list_scholarships         │
              └──────────┬───────────────────────┘
                         │ retrieves from
                         ▼
              ┌──────────────────────────────────┐
              │ ChromaDB (two collections)       │
              │   • universities  (13 pages)     │
              │   • scholarships  (8 pages)      │
              └──────────────────────────────────┘
                         ▲
                         │ built by
              ┌──────────┴───────────────────────┐
              │ ingest.py: fetch MDX from        │
              │ github.com/hashemkhodor/         │
              │   collegesaurus@main             │
              └──────────────────────────────────┘
```

The agent has four tools and decides which to call. Two retrieval tools do semantic search against separate collections so the agent can pick the right domain; two listing tools let it answer "what do you know about?" without a vector search.

## Run locally

```bash
# 1. Get a free Gemini API key
#    https://aistudio.google.com/app/apikey

# 2. Setup
git clone git@github.com:<you>/collegesaurus-ai.git
cd collegesaurus-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then paste your key into .env

# 3. Build the vector index (~30s, runs once)
python ingest.py

# 4. Launch the chat
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public or private — Streamlit Cloud connects to both).
2. Go to https://share.streamlit.io, click **New app**, pick this repo, entry point `app.py`.
3. Under **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
4. First boot will fail because there's no index — SSH in (or run the "Run ingest" button if you add one) and execute `python ingest.py`. Alternatively, run `python ingest.py` locally and commit the resulting `chroma_db/` directory (small — a few MB for this dataset).

Cost at typical traffic: **$0/month**. Gemini 2.0 Flash has a generous free tier (1500 req/day as of writing), and Chroma is embedded.

## Files

| File | Purpose |
| ---- | ------- |
| `app.py` | Streamlit chat UI |
| `agent.py` | Builds the `ReActAgent` with tools and system prompt |
| `tools.py` | The four agent tools (two retrieval, two listing) |
| `ingest.py` | Fetches MDX from the Docusaurus repo, chunks, embeds, persists |
| `config.py` | Shared paths, model names, and environment plumbing |
| `requirements.txt` | Pinned deps |
| `.streamlit/config.toml` | Theme matches the main site (Lebanese green) |

## Updating the knowledge base

When the Docusaurus repo changes, re-run `python ingest.py` — it drops and rebuilds both Chroma collections from scratch. A raw-MDX cache at `source_cache/` is kept for fast re-runs; `rm -rf source_cache/` if you want to force-refetch.

## Integrating with the Docusaurus site

Not in this repo — the Docusaurus side adds a floating chat bubble (bottom-right on every page) that opens an iframe of the Streamlit app in embed mode (`?embed=true`). Handled separately in the `collegesaurus` repo.
