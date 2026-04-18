# Collegesaurus AI

Agentic-RAG chatbot for [Collegesaurus](https://hashemkhodor.github.io/collegesaurus) — answers questions about Lebanese universities and external scholarships using the site's own MDX pages as the knowledge base.

Stack:
- **Streamlit** for the chat UI
- **LlamaIndex** `ReActAgent` for the agent loop
- **ChromaDB** (embedded) for the vector store
- **Gemini 2.5 Flash-Lite** for generation (pay-as-you-go — ~$0.10 / 1M input, $0.40 / 1M output at time of writing)
- **`BAAI/bge-small-en-v1.5`** via fastembed for embeddings (local ONNX, zero API cost, ~130 MB)

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
4. First boot will fail because there's no index — commit the `chroma_db/` directory after running `python ingest.py` locally, or add a `scripts/build-index.sh` that runs it in the Streamlit Cloud container. The dataset is small (~2 MB of vectors), so committing the DB is the lowest-friction option.

> **Billing**: Gemini 2.0 Flash-Lite is pay-as-you-go. Enable billing on the Google Cloud project tied to your API key (https://console.cloud.google.com/billing) — even the free tier requires a billing account now. Cost scales roughly at **$0.30 per 1,000 user messages** at typical RAG prompt sizes.

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
