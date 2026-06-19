# 🚀 Deployment Guide

Your repo started as a terminal app (`chatbot.py`). To deploy it so anyone can
use it in a browser, this adds a **Streamlit web UI** (`app.py`) that reuses all
your original RAG logic. Below is the fastest way to get it live.

---

## Option A — Streamlit Community Cloud (recommended, free)

Best fit for a Python RAG app. ~5 minutes, no server config, deploys straight
from GitHub.

### 1. Push the new files to GitHub
```bash
git add app.py requirements.txt .streamlit/secrets.toml.example .env.example DEPLOY.md .gitignore
git commit -m "Add Streamlit web UI for deployment"
git push
```

### 2. Deploy
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **New app** → pick `amanparganiha/PdfRagAgent`, branch `main`,
   main file `app.py`.
3. Before clicking deploy, open **Advanced settings → Secrets** and paste:
   ```toml
   OPENAI_API_KEY = "sk-proj-your-actual-key-here"
   MODEL = "gpt-4o"
   ```
4. Click **Deploy**. You'll get a public URL like
   `https://pdfragagent.streamlit.app`.

That's it. Users upload PDFs in the sidebar and chat in the main panel.

> **Note on persistence:** Streamlit Cloud has an *ephemeral* filesystem, so
> uploaded PDFs and chat history live in the browser session only and reset when
> the app restarts. That's why `app.py` keeps everything in `st.session_state`
> instead of writing to `memory/` like the CLI does. For permanent storage you'd
> add a managed vector DB (Pinecone, Qdrant, Supabase pgvector) — see below.

---

## Option B — Render / Railway (if you want an always-on container)

1. Add a start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
2. Set the `OPENAI_API_KEY` and `MODEL` environment variables in the dashboard.
3. Point it at this repo; it installs `requirements.txt` automatically.

These give you a longer-lived instance but still ephemeral disk on the free
tiers — same persistence caveat applies.

---

## Running locally (to test before deploying)

```bash
pip install -r requirements.txt

# Option 1: copy the env template
cp .env.example .env          # then paste your key into .env

# Option 2: use Streamlit secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # paste key

streamlit run app.py
```
Opens at http://localhost:8501.

The original CLI still works too: `python chatbot.py`.

---

## Making storage permanent (optional upgrade)

The current deploy re-embeds PDFs each session because cloud disk is wiped on
restart. To persist embeddings and chat across restarts and users, swap the
in-memory `chunks` list for a hosted vector database:

- **Qdrant Cloud** / **Pinecone** — store chunk embeddings, query by similarity
- **Supabase (pgvector)** — embeddings + chat history in Postgres

This is the natural next step once the demo is working, but it's not required to
get a live, shareable app today.
