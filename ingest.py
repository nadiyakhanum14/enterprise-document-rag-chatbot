"""
ingest.py — PDF ingestion pipeline for the Enterprise RAG Chatbot.

Pipeline:
  PDF files → text extraction (page-by-page) → cleaning → chunking
  → embedding (sentence-transformers) → ChromaDB

Design decisions:
  - pypdf for extraction: lightweight, handles most enterprise PDFs without Java
  - RecursiveCharacterTextSplitter: splits on paragraph → sentence → word boundaries
    preserving semantic coherence better than fixed-size splits
  - chunk_size=900, overlap=120: balances context density against LLM token budget
  - Metadata (source_file, page_number) stored with every chunk so citations are exact
  - Idempotent: skips re-indexing if the collection already contains documents

Run:
  python ingest.py
  python ingest.py --reset    # wipe and re-index from scratch
"""

import os
import re
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

DOCS_PATH = os.getenv("DOCS_PATH", "./docs")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking configuration — adjust here to experiment with different strategies
CHUNK_SIZE = 900       # characters per chunk
CHUNK_OVERLAP = 120    # overlap between consecutive chunks


# Ingestion statistics storage
LAST_INGESTION_STATS = {
    "pdfs_discovered": 0,
    "total_pages": 0,
    "pages_normal": 0,
    "pages_ocr": 0,
    "pages_failed": 0,
}

_RAPIDOCR_ENGINE = None


def _get_rapidocr_engine():
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPIDOCR_ENGINE = RapidOCR()
        except ImportError:
            _RAPIDOCR_ENGINE = False
    return _RAPIDOCR_ENGINE if _RAPIDOCR_ENGINE is not False else None


def _find_tesseract_cmd():
    """Check system PATH and standard Windows installation paths for tesseract binary."""
    import shutil
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd

    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    for p in candidate_paths:
        p_path = Path(p)
        if p_path.is_file():
            return str(p_path)
    return None


def ocr_page(pdf_path: Path, page_idx: int) -> str:
    """
    Render a single PDF page to an image and extract text using OCR.
    CPU-friendly: renders at scale 1.5 (~150 DPI), one page at a time.
    Prefers pytesseract if Tesseract is installed, otherwise falls back to RapidOCR (ONNX).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print("  [WARN] pypdfium2 is not installed. Run: pip install pypdfium2")
        return ""

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        if page_idx >= len(pdf):
            return ""
        page = pdf[page_idx]
        pil_image = page.render(scale=1.5).to_pil()
    except Exception as exc:
        print(f"  [WARN] Could not render page {page_idx + 1} of {pdf_path.name}: {exc}")
        return ""

    # Strategy 1: Prefer pytesseract if Tesseract binary is available
    tesseract_cmd = _find_tesseract_cmd()
    if tesseract_cmd:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            text = pytesseract.image_to_string(pil_image)
            if text and text.strip():
                return text
        except Exception as exc:
            print(f"  [DEBUG] pytesseract attempt failed: {exc}")

    # Strategy 2: CPU-friendly ONNX fallback (RapidOCR)
    rapidocr = _get_rapidocr_engine()
    if rapidocr is not None:
        try:
            import numpy as np
            res, _ = rapidocr(np.array(pil_image))
            if res:
                return "\n".join(line[1] for line in res)
        except Exception as exc:
            print(f"  [WARN] RapidOCR execution failed: {exc}")

    # Informative message if no OCR engine is available
    if not tesseract_cmd and rapidocr is None:
        print(
            "  [ERROR] No OCR engine available.\n"
            "          1) For Tesseract: Install Tesseract for Windows:\n"
            "             - Download installer from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "             - Or run: winget install UB-Mannheim.TesseractOCR\n"
            "             - And install python package: pip install pytesseract\n"
            "          2) Or install RapidOCR (pure CPU ONNX): pip install rapidocr_onnxruntime"
        )
    return ""


def clean_text(text: str) -> str:
    """
    Remove excessive whitespace while preserving paragraph structure.
    Scanned PDFs often have artifacts like multiple spaces and stray newlines.
    """
    # Collapse multiple spaces to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than 2 consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(docs_path: str) -> list:
    """
    Load all PDFs from docs_path page-by-page using pypdf.
    Falls back to OCR for pages with no extractable text stream.
    Returns a flat list of dicts: {text, source_file, page_number}
    """
    docs_path = Path(docs_path)
    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{docs_path}'.")
        print("        Add PDF documents to the docs/ folder and run again.")
        return []

    pages = []
    total_pages = 0
    pages_normal = 0
    pages_ocr = 0
    pages_failed = 0

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        try:
            reader = PdfReader(str(pdf_path))
            file_page_count = len(reader.pages)
            total_pages += file_page_count

            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                text = clean_text(raw_text)

                # If normal extraction produces meaningful text (>= 20 chars), keep it
                if len(text) >= 20:
                    pages_normal += 1
                else:
                    # Trigger OCR fallback for scanned/outlined pages
                    print(f"    Page {page_idx + 1}: no text stream found, running OCR fallback...")
                    ocr_raw = ocr_page(pdf_path, page_idx)
                    text = clean_text(ocr_raw)
                    if text:
                        pages_ocr += 1
                    else:
                        pages_failed += 1
                        continue

                pages.append({
                    "text": text,
                    "source_file": pdf_path.name,
                    "page_number": page_idx + 1,  # 1-indexed for human readability
                })

        except Exception as exc:
            print(f"  [WARN] Could not read {pdf_path.name}: {exc}")

    LAST_INGESTION_STATS.update({
        "pdfs_discovered": len(pdf_files),
        "total_pages": total_pages,
        "pages_normal": pages_normal,
        "pages_ocr": pages_ocr,
        "pages_failed": pages_failed,
    })

    print("\n" + "-" * 40)
    print("  Ingestion Page Statistics:")
    print(f"  PDFs discovered          : {len(pdf_files)}")
    print(f"  Total pages              : {total_pages}")
    print(f"  Pages extracted normally : {pages_normal}")
    print(f"  Pages processed with OCR : {pages_ocr}")
    print(f"  Pages with failed extraction: {pages_failed}")
    print("-" * 40 + "\n")
    return pages


def get_ingestion_stats() -> dict:
    """Return a copy of the latest ingestion statistics."""
    return dict(LAST_INGESTION_STATS)


def chunk_documents(pages: list) -> list:
    """
    Split pages into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

    Why RecursiveCharacterTextSplitter?
      It tries paragraph → sentence → word boundaries in order, so chunks are
      more semantically coherent than a naive fixed-size split.

    Returns a list of LangChain Document objects with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        doc = Document(
            page_content=page["text"],
            metadata={
                "source_file": page["source_file"],
                "page_number": page["page_number"],
            },
        )
        page_chunks = splitter.split_documents([doc])
        chunks.extend(page_chunks)

    print(f"  Chunks created: {len(chunks)}")
    print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars")
    return chunks


def build_vectorstore(chunks: list, reset: bool = False):
    """
    Embed chunks with sentence-transformers and store in ChromaDB.
      - 22M parameters — fast on CPU
      - 384-dimensional embeddings — reasonable quality/speed tradeoff
      - Trained on diverse text, handles enterprise document language well

    Why ChromaDB?
      - Embedded (no separate server process needed for development)
      - Persistent storage with simple Python API
      - Good enough for thousands of document chunks on a laptop
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import chromadb
    import shutil

    print(f"\n  Embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    chroma_path = Path(CHROMA_DB_PATH)

    # Handle reset
    if reset and chroma_path.exists():
        shutil.rmtree(chroma_path)
        print("  [INFO] Existing ChromaDB wiped.")

    # Check if already indexed
    if not reset and chroma_path.exists():
        client = chromadb.PersistentClient(path=str(chroma_path))
        try:
            col = client.get_collection("enterprise_docs")
            existing_count = col.count()
            if existing_count > 0:
                print(f"  [INFO] ChromaDB already contains {existing_count} chunks.")
                print("         Use --reset to re-index from scratch.")
                return
        except Exception:
            pass  # Collection doesn't exist yet, proceed normally

    print("  Building ChromaDB vectorstore (this may take a few minutes)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_path),
        collection_name="enterprise_docs",
    )

    # Verify
    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_collection("enterprise_docs")
    print(f"  Indexed chunks: {col.count()}")
    print(f"  ChromaDB path : {chroma_path.resolve()}")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing ChromaDB and re-index from scratch.",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Enterprise RAG — Document Ingestion")
    print("=" * 50)

    print("\n[1/3] Loading PDFs...")
    pages = load_documents(DOCS_PATH)
    if not pages:
        sys.exit(1)

    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(pages)

    print("\n" + "=" * 50)
    print("  Ingestion Summary Statistics:")
    print(f"  PDFs discovered: {LAST_INGESTION_STATS['pdfs_discovered']}")
    print(f"  Total pages: {LAST_INGESTION_STATS['total_pages']}")
    print(f"  Pages extracted normally: {LAST_INGESTION_STATS['pages_normal']}")
    print(f"  Pages processed with OCR: {LAST_INGESTION_STATS['pages_ocr']}")
    print(f"  Pages with failed extraction: {LAST_INGESTION_STATS['pages_failed']}")
    print(f"  Total chunks created: {len(chunks)}")
    print("=" * 50)

    print("\n[3/3] Embedding and indexing...")
    build_vectorstore(chunks, reset=args.reset)

    print("\n[SUCCESS] Ingestion complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
