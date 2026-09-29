"""
chain.py — RAG generation pipeline using OpenRouter as the LLM backend.

Pipeline:
  question + conversation_history
      │
      ├─ HybridRetriever.retrieve()
      │       returns top_k re-ranked Document chunks
      │
      ├─ format_context()
      │       builds a numbered reference list with source + page citations
      │
      ├─ RAG prompt
      │       system: strict grounding instructions + context
      │       human:  conversation history + current question
      │
      └─ OpenRouter LLM → answer + cited sources

Design decisions:
  - Temperature 0.1: factual document QA benefits from low temperature.
    Higher temperature increases creativity but also hallucination risk.
  - Strong refusal instruction: the LLM is explicitly told to refuse rather
    than guess when context is insufficient. This is testable behavior.
  - Conversation history: last N turns are prepended to the human message
    so follow-up questions work correctly without re-retrieving old context.
  - Page numbers come from metadata stored during ingestion — never invented.
"""

import os
import re
from typing import List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct")
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# ── Prompt template ───────────────────────────────────────────────────────────
# Kept as a module-level constant so it can be reviewed, tested, and adjusted
# without touching the rest of the pipeline logic.

SYSTEM_PROMPT = """\
You are an enterprise document assistant. Your job is to answer questions \
based strictly on the document excerpts provided below.

RULES — follow these without exception:
1. Answer ONLY using information that is explicitly present in the provided context.
2. Do NOT use outside knowledge, training data, or assumptions.
3. If the provided context does not contain enough information to answer the \
question, respond with exactly:
   "I don't have enough information in the provided documents to answer that."
4. Do NOT guess, infer beyond what is stated, or fabricate details.
5. Keep answers concise and factual.
6. At the end of every answer that uses document content, list the sources \
in this format:
   Sources:
   - <filename> — Page <number>
   If multiple chunks support the answer, list all relevant sources.

---
DOCUMENT CONTEXT:
{context}
---
"""


REFUSAL_RESPONSE = "I don't have enough information in the provided documents to answer that."


def is_refusal(answer: str) -> bool:
    """
    Check if the model's answer is a refusal indicating the question cannot be
    answered from the provided document context.

    Must not trigger if the answer provides substantive information or cites sources,
    even if it quotes the system's refusal prompt or requirements (e.g. FR-11).
    """
    cleaned = answer.strip()
    if not cleaned:
        return False

    ans_lower = cleaned.lower()

    # If the answer quotes a requirement or provides multiple lines with citations,
    # it is a substantive answer rather than a refusal
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    if len(lines) > 2 and ("system shall" in ans_lower or "fr-11" in ans_lower or "requirements" in ans_lower):
        return False

    refusal_phrases = [
        "don't have enough information",
        "not available in the provided",
        "cannot find",
        "not mentioned",
        "no information",
        "not in the provided",
        "i couldn't find",
    ]

    # Pre-sources portion of answer
    pre_sources = cleaned.split("Sources:")[0].split("sources:")[0].strip().lower()

    for phrase in refusal_phrases:
        if phrase in pre_sources:
            # If it starts with common refusal phrases or is a concise refusal statement:
            if any(pre_sources.startswith(prefix) for prefix in [
                "i don't have", "that information", "i cannot find", "there is no", "not available", "this information"
            ]):
                return True
            if len(pre_sources) < 200:
                return True

    return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_context(docs: List[Document]) -> str:
    """
    Build a numbered context block from retrieved chunks.
    Each chunk includes its source filename and page number.
    """
    if not docs:
        return "No relevant document sections were found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page_number", "?")
        parts.append(f"[{i}] {source} — Page {page}\n{doc.page_content.strip()}")

    return "\n\n".join(parts)


def format_history(history: List[dict]) -> str:
    """
    Convert conversation history list to a plain-text block.
    history format: [{"role": "user"|"assistant", "content": "..."}]
    Only the most recent MAX_HISTORY_TURNS turns are included.
    """
    if not history:
        return ""

    recent = history[-(MAX_HISTORY_TURNS * 2):]  # each turn = 2 entries (user+assistant)
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


def _check_api_key() -> Optional[str]:
    """Return an error message if the API key is not configured."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        return (
            "OpenRouter API key is not configured. "
            "Set OPENROUTER_API_KEY in your .env file."
        )
    return None


def build_llm():
    """Construct the LangChain ChatOpenAI client pointed at OpenRouter."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=1024,
        default_headers={
            # OpenRouter requires these headers to identify the application
            "HTTP-Referer": "https://enterprise-rag.internal",
            "X-Title": "Enterprise Document RAG Chatbot",
        },
    )


def extract_citations(
    answer: str,
    known_files: Optional[List[str]] = None,
) -> List[Tuple[str, int]]:
    """
    Extract (source_file, page_number) tuples explicitly cited in the answer.
    Handles:
      - Sources: - <filename> — Page <number>
      - <filename> - Page <number>
      - <filename>, Pages 1, 2
      - Inline citations: [<filename>, Page <number>], [<filename>, p. <number>]
      - Filenames with or without extension if in known_files
    """
    results: List[Tuple[str, int]] = []
    known_files = known_files or []

    # 1. Standard pattern: filename with extension followed by page number(s)
    pattern = re.compile(
        r'(?:^|[\s\-\*\•\d\.\(\[]+)'
        r'([a-zA-Z0-9_\.\- ]+?\.[a-zA-Z0-9]+)'
        r'\s*(?:[—\–\-,:]|\bat\b)?\s*'
        r'(?:pages?|pp?\.?|pgs?\.?)\s*'
        r'(\d+(?:\s*[\,\-–]\s*\d+)*)',
        re.IGNORECASE | re.MULTILINE,
    )
    for m in pattern.finditer(answer):
        fname = m.group(1).strip().strip("-*•[](), ")
        pages_str = m.group(2).strip()
        for part in re.split(r"[,]", pages_str):
            part = part.strip()
            if "-" in part or "–" in part:
                rp = re.split(r"[-–]", part)
                try:
                    for p in range(int(rp[0].strip()), int(rp[1].strip()) + 1):
                        results.append((fname, p))
                except ValueError:
                    pass
            else:
                try:
                    results.append((fname, int(part)))
                except ValueError:
                    pass

    # 2. Known filenames matching (including stems without extension)
    for kf in known_files:
        stem = kf.rsplit(".", 1)[0]
        escaped_kf = re.escape(kf)
        escaped_stem = re.escape(stem)
        p_known = re.compile(
            rf'(?:{escaped_kf}|{escaped_stem})\s*(?:[—\–\-,:]|\bat\b)?\s*(?:pages?|pp?\.?|pgs?\.?)\s*(\d+)',
            re.IGNORECASE,
        )
        for m in p_known.finditer(answer):
            try:
                p = int(m.group(1))
                if (kf, p) not in results:
                    results.append((kf, p))
            except ValueError:
                pass

    return results


def resolve_sources(
    answer: str,
    docs: List[Document],
    retriever=None,
    conversation_history: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Resolve and return only the source entries that support the generated answer.

    Rules:
      1. If the answer is a refusal, return [] (no citations).
      2. If the answer explicitly cites specific source files and page numbers:
         - Match against retrieved `docs` first (preferred context).
         - If not in `docs`, check conversation history.
         - If not in history, check retriever's stored corpus/vectorstore.
         - Do NOT fabricate or infer citations for unsupported page numbers.
      3. If no explicit citations are found in the answer (fallback):
         - Match chunks from `docs` that have lexical overlap supporting the answer.
         - Fall back to the top-ranked retrieved chunk if no specific overlap exceeds threshold.
      4. All returned sources are deduplicated by (source_file, page_number).
    """
    if is_refusal(answer):
        return []

    # Collect known file names from all available context pools
    known_files: Set[str] = set()
    for doc in docs:
        sf = doc.metadata.get("source_file")
        if sf:
            known_files.add(sf)

    if conversation_history:
        for msg in conversation_history:
            for s in msg.get("sources", []):
                sf = s.get("source_file")
                if sf:
                    known_files.add(sf)

    if retriever and hasattr(retriever, "_corpus_metas") and retriever._corpus_metas:
        for meta in retriever._corpus_metas:
            sf = meta.get("source_file")
            if sf:
                known_files.add(sf)

    cited_pairs = extract_citations(answer, known_files=list(known_files))

    sources: List[dict] = []
    seen: Set[str] = set()

    if cited_pairs:
        for cited_file, cited_page in cited_pairs:
            key = f"{cited_file.lower()}::{cited_page}"
            if key in seen:
                continue

            matched_source: Optional[dict] = None

            # Pool A: Check currently retrieved docs (highest priority)
            for doc in docs:
                sf = doc.metadata.get("source_file", "")
                pn = doc.metadata.get("page_number", "")
                if (
                    sf.lower() == cited_file.lower()
                    or sf.lower().startswith(cited_file.lower())
                    or cited_file.lower().startswith(sf.lower().rsplit(".", 1)[0])
                ) and str(pn) == str(cited_page):
                    matched_source = {
                        "source_file": sf,
                        "page_number": doc.metadata.get("page_number"),
                        "snippet": doc.page_content[:250].strip() + "…",
                    }
                    break

            # Pool B: Check conversation history (e.g. for follow-up questions)
            if not matched_source and conversation_history:
                for msg in reversed(conversation_history):
                    for s in msg.get("sources", []):
                        sf = s.get("source_file", "")
                        pn = s.get("page_number", "")
                        if (
                            sf.lower() == cited_file.lower()
                            or sf.lower().startswith(cited_file.lower())
                            or cited_file.lower().startswith(sf.lower().rsplit(".", 1)[0])
                        ) and str(pn) == str(cited_page):
                            matched_source = {
                                "source_file": s.get("source_file", sf),
                                "page_number": s.get("page_number", pn),
                                "snippet": s.get("snippet", ""),
                            }
                            break
                    if matched_source:
                        break

            # Pool C: Check retriever's loaded corpus or vectorstore
            if not matched_source and retriever:
                if (
                    hasattr(retriever, "_corpus_metas")
                    and hasattr(retriever, "_corpus_texts")
                    and retriever._corpus_metas
                ):
                    for meta, text in zip(retriever._corpus_metas, retriever._corpus_texts):
                        sf = meta.get("source_file", "")
                        pn = meta.get("page_number", "")
                        if (
                            sf.lower() == cited_file.lower()
                            or sf.lower().startswith(cited_file.lower())
                            or cited_file.lower().startswith(sf.lower().rsplit(".", 1)[0])
                        ) and str(pn) == str(cited_page):
                            matched_source = {
                                "source_file": meta.get("source_file", sf),
                                "page_number": meta.get("page_number", pn),
                                "snippet": text[:250].strip() + "…",
                            }
                            break
                elif hasattr(retriever, "vectorstore"):
                    try:
                        matches = retriever.vectorstore.get(
                            where={"$and": [{"source_file": cited_file}, {"page_number": cited_page}]},
                            limit=1,
                        )
                        if matches and matches.get("documents"):
                            text = matches["documents"][0]
                            meta = matches["metadatas"][0] if matches.get("metadatas") else {}
                            matched_source = {
                                "source_file": meta.get("source_file", cited_file),
                                "page_number": meta.get("page_number", cited_page),
                                "snippet": text[:250].strip() + "…",
                            }
                    except Exception:
                        pass

            if matched_source:
                seen.add(key)
                sources.append(matched_source)

    # Fallback: if no citations could be parsed from the answer, select supporting docs
    if not sources and docs:
        answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
        scored_docs = []
        for doc in docs:
            doc_words = set(re.findall(r"\b\w{4,}\b", doc.page_content.lower()))
            overlap = len(answer_words & doc_words)
            scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        candidate_docs = [doc for overlap, doc in scored_docs if overlap >= 3]
        if not candidate_docs and docs:
            candidate_docs = [docs[0]]

        for doc in candidate_docs:
            sf = doc.metadata.get("source_file", "Unknown")
            pn = doc.metadata.get("page_number", "?")
            key = f"{sf}::{pn}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source_file": sf,
                    "page_number": pn,
                    "snippet": doc.page_content[:250].strip() + "…",
                })

    return sources


# ── Main entry point ──────────────────────────────────────────────────────────

def ask(
    question: str,
    retriever,
    conversation_history: Optional[List[dict]] = None,
) -> dict:
    """
    Run the full RAG pipeline for a question.

    Args:
        question: The user's current question.
        retriever: An instance of HybridRetriever.
        conversation_history: List of previous {"role", "content"} dicts.
                              Used so follow-up questions resolve correctly.

    Returns:
        {
            "answer": str,
            "sources": [{"source_file": str, "page_number": int, "snippet": str}],
            "retrieved_chunks": int,
        }
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Guard: check API key before making any LLM call
    key_error = _check_api_key()
    if key_error:
        return {"answer": key_error, "sources": [], "retrieved_chunks": 0}

    # Guard: check for empty question
    question = question.strip()
    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "retrieved_chunks": 0,
        }

    # Stage 1: Retrieve relevant chunks
    # For follow-up questions with pronouns or short queries, incorporate recent user query context
    retrieval_query = question
    if conversation_history:
        recent_user_msgs = [m["content"] for m in conversation_history if m.get("role") == "user"]
        if recent_user_msgs:
            tokens = set(re.findall(r"\b\w+\b", question.lower()))
            pronouns = {"he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "this", "that", "these", "those"}
            if len(tokens) <= 7 or (tokens & pronouns):
                retrieval_query = f"{recent_user_msgs[-1]} {question}"

    try:
        docs = retriever.retrieve(retrieval_query)
    except Exception as exc:
        return {
            "answer": f"Retrieval failed: {exc}",
            "sources": [],
            "retrieved_chunks": 0,
        }

    context = format_context(docs)

    # Stage 2: Build prompt with optional conversation history
    history_text = format_history(conversation_history or [])
    human_message = question
    if history_text:
        human_message = f"Conversation so far:\n{history_text}\n\nCurrent question: {question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Stage 3: Generate answer
    try:
        llm = build_llm()
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": human_message,
        })
    except Exception as exc:
        error_msg = str(exc)
        # Avoid leaking API key or internal details in the UI
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return {
                "answer": "Authentication failed. Check your OPENROUTER_API_KEY in .env.",
                "sources": [],
                "retrieved_chunks": len(docs),
            }
        return {
            "answer": f"Unable to generate a response. Please try again. ({type(exc).__name__})",
            "sources": [],
            "retrieved_chunks": len(docs),
        }

    # Stage 4: Resolve sources supporting the answer
    sources = resolve_sources(
        answer=answer,
        docs=docs,
        retriever=retriever,
        conversation_history=conversation_history,
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(docs),
    }
