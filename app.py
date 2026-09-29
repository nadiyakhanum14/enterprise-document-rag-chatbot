"""
app.py — Streamlit UI for the Enterprise Document RAG Chatbot.

Run:
    streamlit run app.py

Features:
    - Multi-turn conversation with memory (configurable history window)
    - PDF upload + in-app ingestion trigger
    - Source citations with expandable snippets
    - Graceful error handling for missing DB, missing API key, LLM failures
    - Clear chat button
    - Model info display
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment before anything else
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom enterprise CSS (Linear/Notion-inspired minimalist aesthetic)
st.markdown("""
<style>
    /* Clean workspace typography & layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    /* Enterprise header card */
    .header-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        background-color: rgba(33, 150, 243, 0.1);
        color: #1976d2;
        border: 1px solid rgba(33, 150, 243, 0.2);
        margin-bottom: 0.5rem;
    }
    /* Status pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background: rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.08);
        margin-right: 0.5rem;
    }
    /* Source citation cards */
    .source-tag {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
        border-radius: 3px;
        background-color: #f1f3f5;
        color: #495057;
        border: 1px solid #dee2e6;
    }
    /* Suggestion chip buttons */
    div[data-testid="stHorizontalBlock"] button {
        font-size: 0.85rem !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DOCS_PATH = Path(os.getenv("DOCS_PATH", "./docs"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def _api_key_configured() -> bool:
    return bool(API_KEY) and API_KEY != "your_openrouter_api_key_here"


def _chroma_info() -> dict:
    if not (CHROMA_DB_PATH.exists() and any(CHROMA_DB_PATH.iterdir())):
        return {"ready": False, "count": 0}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        col = client.get_collection("enterprise_documents")
        return {"ready": True, "count": col.count()}
    except Exception:
        return {"ready": True, "count": "Ready"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
chroma_status = _chroma_info()

with st.sidebar:
    st.markdown('<span class="header-badge">ENTERPRISE INTELLIGENCE</span>', unsafe_allow_html=True)
    st.title("Document RAG")
    st.caption("Grounded knowledge retrieval for enterprise document assets.")
    st.divider()

    # System status indicators
    st.subheader("System Infrastructure")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if _api_key_configured():
            st.success("API Connected", icon="🟢")
        else:
            st.error("API Missing", icon="🔴")
    with col_s2:
        if chroma_status["ready"]:
            st.success(f"{chroma_status['count']} Chunks", icon="📚")
        else:
            st.warning("No Index", icon="⚠️")

    st.caption(f"**Model:** `{LLM_MODEL}`")
    st.caption(f"**Embeddings:** `all-MiniLM-L6-v2` (384d)")
    st.caption(f"**Reranker:** `ms-marco-MiniLM-L-6-v2`")
    st.divider()

    # Document upload
    st.subheader("Document Repository")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload enterprise PDFs. Scanned/raster pages automatically trigger OCR.",
    )

    if uploaded_files:
        DOCS_PATH.mkdir(exist_ok=True)
        saved = []
        for f in uploaded_files:
            dest = DOCS_PATH / f.name
            dest.write_bytes(f.read())
            saved.append(f.name)
        if saved:
            st.success(f"Uploaded: {', '.join(saved)}")

    if st.button("🔄 Ingest & Re-index Knowledge Base", type="primary", use_container_width=True):
        if not any(DOCS_PATH.glob("*.pdf")):
            st.error("No PDF files found in docs/ directory.")
        else:
            with st.spinner("Extracting text, chunking, and embedding into ChromaDB…"):
                try:
                    if str(Path(__file__).parent) not in sys.path:
                        sys.path.insert(0, str(Path(__file__).parent))

                    from ingest import load_documents, chunk_documents, build_vectorstore
                    pages = load_documents(str(DOCS_PATH))
                    if pages:
                        chunks = chunk_documents(pages)
                        build_vectorstore(chunks, reset=True)
                        st.success(f"✓ Indexed {len(chunks)} chunks from {len(pages)} pages.")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error("No text could be extracted from the PDFs.")
                except Exception as exc:
                    st.error(f"Ingestion error: {exc}")

    st.divider()

    # Settings
    st.subheader("Retrieval Controls")
    top_k = st.slider(
        "Context Chunks (Top-K)",
        min_value=2,
        max_value=10,
        value=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        help="Number of re-ranked context chunks provided to the LLM.",
    )
    show_sources = st.toggle("Display Source Citations", value=True)

    st.divider()
    st.caption("Hybrid Search: BM25 + Semantic Vector + Cross-Encoder Rerank")


# ── Load retriever (cached per top_k value) ───────────────────────────────────
@st.cache_resource(show_spinner="Initializing hybrid retrieval models…")
def get_retriever(top_k: int):
    if not chroma_status["ready"]:
        return None
    try:
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))
        from retriever import HybridRetriever
        return HybridRetriever(top_k=top_k)
    except Exception as exc:
        st.error(f"Failed to load retriever: {exc}")
        return None


# ── Main UI ───────────────────────────────────────────────────────────────────
col_title, col_clear = st.columns([5, 1])
with col_title:
    st.markdown('<span class="header-badge">ENTERPRISE KNOWLEDGE ASSISTANT</span>', unsafe_allow_html=True)
    st.title("Document Intelligence Chatbot")
    st.caption("Production-grade grounded document Q&A with hybrid retrieval, cross-encoder reranking, and citation verification.")

with col_clear:
    st.write("")
    st.write("")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Pre-flight warnings in main area
if not _api_key_configured():
    st.error(
        "**API Key Required:** Please configure `OPENROUTER_API_KEY` in `.env` to enable LLM generation."
    )

if not chroma_status["ready"]:
    st.warning(
        "**Knowledge Base Unindexed:** Please upload PDFs in the sidebar and click **Ingest & Re-index Knowledge Base**."
    )

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick-start suggestion chips when chat is empty
if len(st.session_state.messages) == 0 and chroma_status["ready"] and _api_key_configured():
    st.markdown("##### 💡 Suggested Questions")
    chips = [
        "What are the core functional requirements?",
        "What does the end-to-end architecture look like?",
        "What are the key rollout phases?",
        "What is the capital of France?",
    ]
    chip_cols = st.columns(len(chips))
    for i, chip_text in enumerate(chips):
        with chip_cols[i]:
            if st.button(chip_text, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_prompt = chip_text
                st.rerun()

# Check for pending prompt from suggestion chips
active_prompt = None
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    del st.session_state.pending_prompt

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (
            msg["role"] == "assistant"
            and show_sources
            and msg.get("sources")
        ):
            with st.expander(f"📎 Verified Citations ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f"📄 **`{s['source_file']}`** &nbsp;•&nbsp; **Page {s['page_number']}**\n\n"
                        f"> {s['snippet']}\n"
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a question grounded in your indexed documents…",
    disabled=not (_api_key_configured() and chroma_status["ready"]),
)

# Handle either typed user_input or active_prompt from chip
query_to_run = active_prompt if active_prompt else user_input

if query_to_run:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query_to_run})
    with st.chat_message("user"):
        st.markdown(query_to_run)

    # Generate response
    with st.chat_message("assistant"):
        retriever = get_retriever(top_k)

        if retriever is None:
            response = {
                "answer": (
                    "Documents have not been indexed yet. "
                    "Please upload PDFs and click **Ingest & Re-index Knowledge Base** in the sidebar."
                ),
                "sources": [],
            }
            st.markdown(response["answer"])
        else:
            with st.spinner("Retrieving relevant context and generating grounded answer…"):
                try:
                    if str(Path(__file__).parent) not in sys.path:
                        sys.path.insert(0, str(Path(__file__).parent))
                    from chain import ask

                    # Pass conversation history (exclude current user message)
                    history = st.session_state.messages[:-1]

                    response = ask(
                        question=query_to_run,
                        retriever=retriever,
                        conversation_history=history,
                    )
                except Exception as exc:
                    response = {
                        "answer": f"An error occurred while generating the answer: {exc}",
                        "sources": [],
                    }

            st.markdown(response["answer"])

            if show_sources and response.get("sources"):
                with st.expander(f"📎 Verified Citations ({len(response['sources'])})", expanded=False):
                    for s in response["sources"]:
                        st.markdown(
                            f"📄 **`{s['source_file']}`** &nbsp;•&nbsp; **Page {s['page_number']}**\n\n"
                            f"> {s['snippet']}\n"
                        )

    # Persist assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
    })
