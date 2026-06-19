"""
🤖 Streamlit web UI for the PDF RAG Agent.

Wraps the core logic (chunking, embeddings, semantic search, web search,
function-calling agent) in a browser interface so it can be deployed.

Key differences from the CLI version (chatbot.py):
- API key comes from Streamlit Secrets / env, not a .env file
- PDFs are uploaded through the browser (no pdfs/ folder needed)
- Chat history + embeddings live in st.session_state, NOT on disk
  (cloud hosts have ephemeral filesystems, so disk persistence is unreliable)
"""

import os
import json
import tempfile

import numpy as np
import streamlit as st
from openai import OpenAI

# ── Optional deps (same graceful degradation as the CLI) ──────────────────
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
def get_api_key() -> str | None:
    """Look in Streamlit Secrets first, then env vars."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_model() -> str:
    try:
        if "MODEL" in st.secrets:
            return st.secrets["MODEL"]
    except Exception:
        pass
    return os.getenv("MODEL", "gpt-4o")


EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful, friendly AI assistant.
You have two superpowers:
1. You can search uploaded PDF documents for specific information.
2. You can search the web for current information.

When answering from documents, cite which PDF the info came from.
If the documents don't contain the answer, say so clearly and offer to search the web.
Be concise but thorough. Use a conversational tone."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search through uploaded PDF documents for relevant information. "
                "Use this when the user asks questions that might be answered by "
                "their uploaded files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find in documents",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this when the user "
                "asks about recent events, real-time data, or anything NOT in "
                "the uploaded documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up online",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT STORE (in-memory; chunks live in session_state)
# ═══════════════════════════════════════════════════════════════════════════
class DocumentStore:
    """PDF chunking, embedding, and cosine-similarity search. No disk cache."""

    def __init__(self, client: OpenAI, chunks: list[dict]):
        self.client = client
        self.chunks = chunks  # shared reference to session_state list

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not PDF_AVAILABLE:
            return ""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()

    def chunk_text(self, text: str, source: str) -> list[dict]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({"text": chunk.strip(), "source": source})
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: list[str], progress=None) -> list[list[float]]:
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([d.embedding for d in response.data])
            if progress is not None:
                progress.progress(min(i + batch_size, len(texts)) / len(texts))
        return all_embeddings

    def load_pdf_from_path(self, pdf_path: str, filename: str, progress=None) -> int:
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return 0
        raw_chunks = self.chunk_text(text, source=filename)
        texts = [c["text"] for c in raw_chunks]
        embeddings = self.get_embeddings_batch(texts, progress=progress)
        for chunk, emb in zip(raw_chunks, embeddings):
            chunk["embedding"] = emb
        self.chunks.extend(raw_chunks)
        return len(raw_chunks)

    def search(self, query: str, top_k: int = TOP_K) -> str:
        if not self.chunks:
            return "No documents loaded. Upload a PDF in the sidebar first."
        query_emb = np.array(self.get_embedding(query))
        scored = []
        for chunk in self.chunks:
            chunk_emb = np.array(chunk["embedding"])
            similarity = np.dot(query_emb, chunk_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-10
            )
            scored.append((similarity, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        results = []
        for i, (score, chunk) in enumerate(top, 1):
            results.append(
                f"[Result {i} | {chunk['source']} | relevance: {score:.2f}]\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(results)


# ═══════════════════════════════════════════════════════════════════════════
# WEB SEARCH
# ═══════════════════════════════════════════════════════════════════════════
def web_search(query: str, max_results: int = 5) -> str:
    if not SEARCH_AVAILABLE:
        return "Web search not available on this deployment."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for: {query}"
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r['title']}\n   {r['body']}\n   Source: {r['href']}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# AGENT (function-calling loop, identical logic to the CLI)
# ═══════════════════════════════════════════════════════════════════════════
class Agent:
    def __init__(self, client: OpenAI, docs: DocumentStore, model: str):
        self.client = client
        self.docs = docs
        self.model = model
        self.tool_handlers = {
            "search_documents": lambda a: self.docs.search(a.get("query", "")),
            "web_search": lambda a: web_search(a.get("query", "")),
        }

    def _active_tools(self):
        tools = []
        if self.docs.chunks:
            tools.append(TOOLS[0])
        if SEARCH_AVAILABLE:
            tools.append(TOOLS[1])
        return tools or None

    def chat(self, history: list[dict]) -> str:
        doc_info = ""
        if self.docs.chunks:
            sources = set(c["source"] for c in self.docs.chunks)
            doc_info = f"\n\nLoaded documents: {', '.join(sources)}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT + doc_info}, *history]
        tools = self._active_tools()

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, temperature=0.7
        )
        msg = response.choices[0].message

        while msg.tool_calls:
            messages.append(msg.model_dump())
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                handler = self.tool_handlers.get(fn_name)
                result = handler(fn_args) if handler else f"Unknown tool: {fn_name}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, temperature=0.7
            )
            msg = response.choices[0].message

        return msg.content or "(no response)"


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Chat with your PDFs", page_icon="🤖", layout="centered")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content}]
if "chunks" not in st.session_state:
    st.session_state.chunks = []            # embedded PDF chunks
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = set()   # filenames already processed

api_key = get_api_key()
model = get_model()

st.title("🤖 Chat with your PDFs")
st.caption("RAG agent powered by OpenAI — upload PDFs, ask questions, or search the web.")

# ── Sidebar: API key + uploads + controls ─────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Set this in Streamlit Secrets for permanent deployment.",
        )
        if not api_key:
            st.warning("Enter an API key to start.")

    st.divider()
    st.header("📄 Documents")

    if not PDF_AVAILABLE:
        st.error("PyPDF2 not installed.")

    uploaded = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if uploaded and api_key:
        client = OpenAI(api_key=api_key)
        store = DocumentStore(client, st.session_state.chunks)
        for uf in uploaded:
            if uf.name in st.session_state.loaded_files:
                continue
            with st.status(f"Processing {uf.name}...", expanded=False) as status:
                bar = st.progress(0.0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name
                try:
                    n = store.load_pdf_from_path(tmp_path, uf.name, progress=bar)
                    if n:
                        st.session_state.loaded_files.add(uf.name)
                        status.update(label=f"✅ {uf.name} — {n} chunks", state="complete")
                    else:
                        status.update(label=f"⚠️ No text in {uf.name}", state="error")
                finally:
                    os.unlink(tmp_path)

    # Show what's loaded
    if st.session_state.loaded_files:
        st.success(f"{len(st.session_state.loaded_files)} file(s) loaded")
        counts = {}
        for c in st.session_state.chunks:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        for src, cnt in counts.items():
            st.write(f"📄 {src} — {cnt} chunks")

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Model: `{model}`")
    st.caption(f"Web search: {'✅' if SEARCH_AVAILABLE else '❌'}")

# ── Main chat area ─────────────────────────────────────────────────────────
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about your PDFs or anything else...")

if prompt:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = OpenAI(api_key=api_key)
    store = DocumentStore(client, st.session_state.chunks)
    agent = Agent(client, store, model)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = agent.chat(st.session_state.messages)
            except Exception as e:
                reply = f"⚠️ Error: {e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
