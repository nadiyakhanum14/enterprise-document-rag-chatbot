"""
api.py — Production FastAPI backend for Enterprise Document RAG Chatbot.

Endpoints:
  GET  /health            — System status, DB readiness, and model info
  POST /query             — Core RAG Q&A endpoint with memory and citation support
  GET  /documents         — List discovered PDF documents and indexed chunk count
  POST /documents/index   — Trigger document ingestion/re-indexing
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# Ensure local modules are importable
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import chain
import ingest
from retriever import HybridRetriever

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
DOCS_PATH = Path(os.getenv("DOCS_PATH", "./docs"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

app = FastAPI(
    title="Enterprise Document RAG Chatbot API",
    description="REST API for grounded document question answering with hybrid retrieval, cross-encoder reranking, and citation tracking.",
    version="1.0.0",
)

# Enable CORS for enterprise frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached retriever instance
_retriever_instance: Optional[HybridRetriever] = None


def get_retriever(top_k: int = DEFAULT_TOP_K) -> Optional[HybridRetriever]:
    """Lazy loader for HybridRetriever with top_k configuration."""
    global _retriever_instance
    if not (CHROMA_DB_PATH.exists() and any(CHROMA_DB_PATH.iterdir())):
        return None
    if _retriever_instance is None or _retriever_instance.top_k != top_k:
        _retriever_instance = HybridRetriever(top_k=top_k)
    return _retriever_instance


# ── Pydantic Request & Response Schemas ────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    api_key_configured: bool = Field(..., examples=[True])
    chroma_ready: bool = Field(..., examples=[True])
    indexed_chunks: int = Field(..., examples=[21])
    llm_model: str = Field(..., examples=["openai/gpt-4o-mini"])


class SourceItem(BaseModel):
    source_file: str = Field(..., examples=["RAGG.pdf"])
    page_number: Any = Field(..., examples=[3])
    snippet: str = Field(..., examples=["Raj values speed and simplicity…"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What does Raj prioritize?"])
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        examples=[
            [
                {"role": "user", "content": "Who is Raj and what does he value?"},
                {"role": "assistant", "content": "Raj is a new hire in Operations who values speed and simplicity."},
            ]
        ],
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20, examples=[5])


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    retrieved_chunks: int


class DocumentItem(BaseModel):
    filename: str
    size_bytes: int


class DocumentsResponse(BaseModel):
    documents_folder: str
    documents: List[DocumentItem]
    total_documents: int
    indexed_chunks: int


class IndexRequest(BaseModel):
    reset: bool = Field(default=False, description="Wipe existing vectorstore before indexing")


class IndexResponse(BaseModel):
    status: str
    indexed_chunks: int
    total_pages: int
    pages_normal: int
    pages_ocr: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Enterprise Document RAG Chatbot API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Verify backend status, ChromaDB vector store, and LLM configuration."""
    api_key_ok = bool(OPENROUTER_API_KEY) and OPENROUTER_API_KEY != "your_openrouter_api_key_here"
    chroma_ok = CHROMA_DB_PATH.exists() and any(CHROMA_DB_PATH.iterdir())

    chunks = 0
    if chroma_ok:
        try:
            r = get_retriever()
            if r is not None:
                chunks = r.get_corpus_size()
        except Exception:
            chunks = 0

    return HealthResponse(
        status="healthy" if (api_key_ok and chroma_ok) else "degraded",
        api_key_configured=api_key_ok,
        chroma_ready=chroma_ok,
        indexed_chunks=chunks,
        llm_model=LLM_MODEL,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(req: QueryRequest):
    """
    Execute grounded RAG generation for an enterprise document question.
    Preserves exact citations, conversation memory, and strict refusal behavior.
    """
    clean_question = req.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or blank.",
        )

    # Verify vector store availability
    retriever = get_retriever(top_k=req.top_k or DEFAULT_TOP_K)
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document index is not initialized. Please ingest documents first.",
        )

    try:
        result = chain.ask(
            question=clean_question,
            retriever=retriever,
            conversation_history=req.conversation_history,
        )
    except Exception as exc:
        # Prevent leaking raw exception details or internal keys
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {type(exc).__name__}",
        )

    return QueryResponse(
        question=clean_question,
        answer=result.get("answer", ""),
        sources=[
            SourceItem(
                source_file=s.get("source_file", "Unknown"),
                page_number=s.get("page_number", "?"),
                snippet=s.get("snippet", ""),
            )
            for s in result.get("sources", [])
        ],
        retrieved_chunks=result.get("retrieved_chunks", 0),
    )


@app.get("/documents", response_model=DocumentsResponse, tags=["Documents"])
def list_documents():
    """List all available PDF documents in docs/ directory and indexed chunk count."""
    docs = []
    if DOCS_PATH.exists():
        for p in sorted(DOCS_PATH.glob("*.pdf")):
            docs.append(DocumentItem(filename=p.name, size_bytes=p.stat().st_size))

    chunk_count = 0
    r = get_retriever()
    if r is not None:
        chunk_count = r.get_corpus_size()

    return DocumentsResponse(
        documents_folder=str(DOCS_PATH),
        documents=docs,
        total_documents=len(docs),
        indexed_chunks=chunk_count,
    )


@app.post("/documents/index", response_model=IndexResponse, tags=["Documents"])
def index_documents(req: IndexRequest):
    """Trigger ingestion pipeline over PDF documents in docs/."""
    global _retriever_instance
    if not DOCS_PATH.exists() or not any(DOCS_PATH.glob("*.pdf")):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No PDF documents found in docs/ directory.",
        )

    try:
        pages = ingest.load_documents(str(DOCS_PATH))
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the documents.",
            )

        chunks = ingest.chunk_documents(pages)
        ingest.build_vectorstore(chunks, reset=req.reset)

        # Invalidate cached retriever so it reloads fresh index
        _retriever_instance = None
        stats = ingest.get_ingestion_stats()

        return IndexResponse(
            status="success",
            indexed_chunks=len(chunks),
            total_pages=stats.get("total_pages", len(pages)),
            pages_normal=stats.get("pages_normal", 0),
            pages_ocr=stats.get("pages_ocr", 0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {type(exc).__name__}",
        )
