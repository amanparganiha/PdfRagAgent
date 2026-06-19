"""
🤖 Conversational AI Agent with PDF RAG
- OpenAI ChatGPT API
- PDF upload & question answering (RAG)
- Persistent memory (saves chat history to disk)
- Web search via DuckDuckGo (no API key needed)
- Function calling so the model decides WHEN to search vs use docs
"""

import os
import json
import datetime
import hashlib
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file (keeps your API key out of the code)
load_dotenv()

# ── PDF reading ──────────────────────────────────────────────────────────
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PDF support disabled. Install it with: pip install PyPDF2")

# ── Web search ───────────────────────────────────────────────────────────
try:
    from duckduckgo_search import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    print("⚠️  Web search disabled. Install it with: pip install duckduckgo-search")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — Put your API key here directly OR use environment variable
# ═══════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found! Create a .env file with your key.")
    print("   See .env.example for reference.")
    exit(1)
MODEL = os.getenv("MODEL", "gpt-4o")
EMBEDDING_MODEL = "text-embedding-3-small"
MEMORY_FILE = Path("memory/chat_history.json")
EMBEDDINGS_DIR = Path("memory/embeddings")
PDF_DIR = Path("pdfs")                  # drop your PDFs here
MAX_HISTORY = 50
CHUNK_SIZE = 800                        # characters per chunk
CHUNK_OVERLAP = 200                     # overlap between chunks
TOP_K = 5                               # number of chunks to retrieve

SYSTEM_PROMPT = """You are a helpful, friendly AI assistant.
You have two superpowers:
1. You can search uploaded PDF documents for specific information.
2. You can search the web for current information.

When answering from documents, cite which PDF the info came from.
If the documents don't contain the answer, say so clearly and offer to search the web.
Be concise but thorough. Use a conversational tone."""


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS DEFINITION (OpenAI function calling)
# ═══════════════════════════════════════════════════════════════════════════
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
# PDF PROCESSOR & VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════
class DocumentStore:
    """Handles PDF loading, chunking, embedding, and semantic search."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.chunks: list[dict] = []
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        PDF_DIR.mkdir(parents=True, exist_ok=True)

    # ── PDF text extraction ───────────────────────────────────────────
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

    # ── Chunking ──────────────────────────────────────────────────────
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

    # ── Embeddings ────────────────────────────────────────────────────
    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
            print(f"   📊 Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        return all_embeddings

    # ── Cache (avoid re-embedding same PDF) ───────────────────────────
    def _cache_path(self, pdf_path: str) -> Path:
        file_hash = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()
        return EMBEDDINGS_DIR / f"{file_hash}.json"

    def _load_cache(self, pdf_path: str) -> list[dict] | None:
        cache = self._cache_path(pdf_path)
        if cache.exists():
            return json.loads(cache.read_text())
        return None

    def _save_cache(self, pdf_path: str, chunks: list[dict]):
        cache = self._cache_path(pdf_path)
        cache.write_text(json.dumps(chunks))

    # ── Load a single PDF ─────────────────────────────────────────────
    def load_pdf(self, pdf_path: str):
        filename = Path(pdf_path).name
        print(f"\n📄 Processing: {filename}")

        cached = self._load_cache(pdf_path)
        if cached:
            self.chunks.extend(cached)
            print(f"   ⚡ Loaded {len(cached)} chunks from cache")
            return

        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"   ⚠️  No text found in {filename}")
            return

        raw_chunks = self.chunk_text(text, source=filename)
        print(f"   ✂️  Split into {len(raw_chunks)} chunks")

        texts = [c["text"] for c in raw_chunks]
        embeddings = self.get_embeddings_batch(texts)

        for chunk, emb in zip(raw_chunks, embeddings):
            chunk["embedding"] = emb

        self.chunks.extend(raw_chunks)
        self._save_cache(pdf_path, raw_chunks)
        print(f"   ✅ Done! ({len(raw_chunks)} chunks indexed)")

    # ── Load all PDFs from the folder ─────────────────────────────────
    def load_all_pdfs(self):
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"\n📁 No PDFs found in '{PDF_DIR}/' folder.")
            print(f"   Drop your PDF files there and restart, or use /load <path>")
            return
        print(f"\n📚 Found {len(pdf_files)} PDF(s) in '{PDF_DIR}/'")
        for pdf in pdf_files:
            self.load_pdf(str(pdf))
        print(f"\n✅ Total: {len(self.chunks)} chunks ready for search\n")

    # ── Semantic search ───────────────────────────────────────────────
    def search(self, query: str, top_k: int = TOP_K) -> str:
        if not self.chunks:
            return "No documents loaded. Use /load <path> to add a PDF."

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
# MEMORY MANAGER
# ═══════════════════════════════════════════════════════════════════════════
class Memory:
    def __init__(self, filepath: Path = MEMORY_FILE, max_messages: int = MAX_HISTORY):
        self.filepath = filepath
        self.max_messages = max_messages
        self.history: list[dict] = []
        self._load()

    def _load(self):
        if self.filepath.exists():
            try:
                self.history = json.loads(self.filepath.read_text())
                print(f"📂 Loaded {len(self.history)} messages from memory")
            except json.JSONDecodeError:
                self.history = []
        else:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def save(self):
        self.filepath.write_text(json.dumps(self.history, indent=2))

    def add(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]
        self.save()

    def get_messages(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.history]

    def clear(self):
        self.history = []
        self.save()
        print("🗑️  Memory cleared!")


# ═══════════════════════════════════════════════════════════════════════════
# WEB SEARCH
# ═══════════════════════════════════════════════════════════════════════════
def web_search(query: str, max_results: int = 5) -> str:
    if not SEARCH_AVAILABLE:
        return "Web search not available. Install: pip install duckduckgo-search"
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
# AGENT
# ═══════════════════════════════════════════════════════════════════════════
class Agent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.memory = Memory()
        self.docs = DocumentStore(self.client)
        self.tool_handlers = {
            "search_documents": self._handle_doc_search,
            "web_search": self._handle_web_search,
        }

    def _handle_doc_search(self, args: dict) -> str:
        query = args.get("query", "")
        print(f"   📄 Searching docs: {query}")
        return self.docs.search(query)

    def _handle_web_search(self, args: dict) -> str:
        query = args.get("query", "")
        print(f"   🔍 Searching web: {query}")
        return web_search(query)

    def _get_active_tools(self) -> list[dict]:
        tools = []
        if self.docs.chunks:
            tools.append(TOOLS[0])  # search_documents
        if SEARCH_AVAILABLE:
            tools.append(TOOLS[1])  # web_search
        return tools or None

    def _build_messages(self) -> list[dict]:
        doc_info = ""
        if self.docs.chunks:
            sources = set(c["source"] for c in self.docs.chunks)
            doc_info = f"\n\nLoaded documents: {', '.join(sources)}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT + doc_info},
            *self.memory.get_messages(),
        ]

    def chat(self, user_input: str) -> str:
        self.memory.add("user", user_input)
        messages = self._build_messages()
        tools = self._get_active_tools()

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            temperature=0.7,
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
                model=MODEL,
                messages=messages,
                tools=tools,
                temperature=0.7,
            )
            msg = response.choices[0].message

        reply = msg.content or "(no response)"
        self.memory.add("assistant", reply)
        return reply

    def load_pdf(self, path: str):
        self.docs.load_pdf(path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  🤖 AI Agent with PDF RAG  |  type 'quit' to exit")
    print("  Commands:")
    print("    /load <path>    — Load a PDF file")
    print("    /docs           — List loaded documents")
    print("    /clear          — Reset chat memory")
    print("    /history        — View recent messages")
    print("=" * 60)

    agent = Agent()

    # Auto-load PDFs from the pdfs/ folder
    agent.docs.load_all_pdfs()

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────
        if user_input.lower() == "quit":
            print("👋 Bye!")
            break

        elif user_input.lower().startswith("/load "):
            pdf_path = user_input[6:].strip().strip('"').strip("'")
            if not Path(pdf_path).exists():
                print(f"❌ File not found: {pdf_path}")
            elif not pdf_path.lower().endswith(".pdf"):
                print("❌ Only PDF files are supported.")
            else:
                agent.load_pdf(pdf_path)
            continue

        elif user_input.lower() == "/docs":
            if not agent.docs.chunks:
                print("📁 No documents loaded.")
            else:
                sources = {}
                for c in agent.docs.chunks:
                    sources[c["source"]] = sources.get(c["source"], 0) + 1
                print("📚 Loaded documents:")
                for src, count in sources.items():
                    print(f"   📄 {src} ({count} chunks)")
            continue

        elif user_input.lower() == "/clear":
            agent.memory.clear()
            continue

        elif user_input.lower() == "/history":
            for m in agent.memory.history[-10:]:
                role = "🧑" if m["role"] == "user" else "🤖"
                ts = m.get("timestamp", "")[:16]
                print(f"  {role} [{ts}] {m['content'][:100]}")
            continue

        # ── Chat ──────────────────────────────────────────────────
        print("\n🤖 Agent: ", end="", flush=True)
        reply = agent.chat(user_input)
        print(reply)


if __name__ == "__main__":
    main()