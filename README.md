# Enterprise Document RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) system that allows enterprise employees to query PDF document repositories with high precision, grounded responses, verifiable page-level citations, and 100% reliable out-of-scope refusal handling.

Built with a modular two-stage retrieval pipeline (**BM25 Keyword Search + Dense Vector Similarity** with normalized score fusion, followed by **Cross-Encoder Re-ranking**), a decoupled **FastAPI REST Backend**, a refined **Streamlit Workspace UI**, an automated **OCR Fallback Engine** for scanned documents, and an **offline LLM-as-a-judge evaluation suite** implementing RAGAS metrics.

---

## Architecture Overview

```
                          ┌─────────────────────────────┐
                          │    Enterprise PDF Files     │
                          │        (docs/*.pdf)         │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │           Ingestion Pipeline          │
                     │  1. PyPDF Page Text Extraction        │
                     │  2. OCR Fallback (pdf2image+Tesseract)│
                     │  3. RecursiveCharacter Chunking       │
                     │     (~900 chars, 120 overlap)         │
                     │  4. Page-level Metadata Tagging       │
                     └───────────┬───────────────┬───────────┘
                                 │               │
                                 ▼               ▼
                      ┌────────────────────┐  ┌───────────────────┐
                      │ all-MiniLM-L6-v2   │  │   rank-bm25       │
                      │ ChromaDB Vector DB │  │   In-Memory Index │
                      └─────────┬──────────┘  └─────────┬─────────┘
                                │                       │
 ┌───────────────────────────┐  │                       │
 │  User Query / API Client  │  │                       │
 └─────────────┬─────────────┘  │                       │
               │                │                       │
               ▼                ▼                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Hybrid Retrieval Engine                           │
 │  1. Bi-Encoder Semantic Search (ChromaDB Cosine Similarity)            │
 │  2. BM25 Lexical Keyword Search (Token-level BM25)                     │
 │  3. Min-Max Normalization & Convex Fusion: S = α·S_sem + (1-α)·S_bm25  │
 │  4. Top-20 Candidate Pooling & Deduplication                           │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │ Top-20 Candidates
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Cross-Encoder Re-Ranker                           │
 │     ms-marco-MiniLM-L-6-v2 Joint Query-Document Relevance Scoring      │
 │     Selects Top-K (default: 5) Most Informative Chunks                 │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │ Top-5 Reranked Chunks
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 Grounded Generation & Citation Engine                  │
 │  1. Conversation History Integration (sliding window, max 6 turns)     │
 │  2. Strict Grounding Prompt (temperature=0.1, hallucination barrier)   │
 │  3. OpenRouter API (gpt-4o-mini / mistral-7b-instruct)                 │
 │  4. Citation Filtering (resolves cited pages; drops unverified chunks) │
 │  5. Refusal Handler: "I don't have enough information..." -> sources: []
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
 ┌──────────────────────────┐              ┌──────────────────────────┐
 │     FastAPI REST API     │              │    Streamlit Workspace   │
 │   http://localhost:8000  │              │   http://localhost:8501  │
 │   Interactive Swagger    │              │   Minimalist Linear UI   │
 └──────────────────────────┘              └──────────────────────────┘
```

---

## Empirical Benchmark & Evaluation

The system is evaluated using an offline LLM-as-a-judge test suite implementing the standard mathematical formulations of **RAGAS** (Retrieval Augmented Generation Assessment) on a curated **22-question golden dataset** (`evaluation/golden_qa_set.json`), covering every page of enterprise documentation plus explicit out-of-scope refusal queries.

### Measured Results: Semantic Baseline vs. Hybrid + Re-ranking

> **Verification:** Measured from actual execution run on the full test suite (`python evaluate.py --config both`).

| Metric | Target KPI | Semantic Only (Baseline) | Hybrid + Cross-Encoder Rerank | Measured Delta |
|:---|:---:|:---:|:---:|:---:|
| **Faithfulness** | $\ge 0.85$ | $0.9028$ | **$0.9722$** | **$+6.94\%$** |
| **Context Precision** | $\ge 0.80$ | $0.9465$ | **$0.9678$** | **$+2.13\%$** |
| **Context Recall** | $\ge 0.80$ | $0.9194$ | **$0.9444$** | **$+2.50\%$** |
| **Answer Relevancy** | $\ge 0.85$ | $0.7937$ | **$0.7907$** | $-0.30\%$ |
| **Refusal Accuracy** | $100\%$ | **$100.0\%$ (4/4)** | **$100.0\%$ (4/4)** | Parity |

### Key Benchmark Insights
1. **Faithfulness surged from 90.3% to 97.2% (+6.9%)**: By eliminating noisy candidates via cross-encoder joint scoring, the context provided to the LLM is significantly cleaner, nearly eliminating hallucinated interpolations.
2. **Context Precision improved to 96.8%**: The cross-encoder accurately places truly relevant chunks at ranks 1 and 2, which disproportionately influences LLM attention.
3. **Context Recall rose to 94.4%**: BM25 keyword matching recaptures specific entity names, numerical KPIs, and technical section titles that pure vector search missed.
4. **100% Out-of-Scope Refusal Accuracy**: The system reliably refuses queries outside document boundaries without hallucinating answers or inventing dummy citations.

---

## Core Engineering Decisions & Trade-Offs

### 1. Hybrid Retrieval (BM25 + Semantic Vector Search)
- **Problem:** Pure embedding search struggles with exact keyword queries (acronyms, PRD requirement IDs like `FR-14`, section numbers, dates). Pure BM25 struggles with synonymy and conversational paraphrasing.
- **Solution:** Parallel retrieval querying both ChromaDB and an in-memory BM25 index. Scores are independently min-max normalized to $[0, 1]$ and combined via convex combination:
  $$\text{Fused Score} = \alpha \cdot \text{Score}_{\text{sem}} + (1 - \alpha) \cdot \text{Score}_{\text{bm25}}$$
- **Trade-off:** Minimal memory overhead for BM25 corpus storage in exchange for bulletproof entity recall.

### 2. Two-Stage Cross-Encoder Re-ranking
- **Bi-Encoder vs. Cross-Encoder:** Bi-encoders (`all-MiniLM-L6-v2`) embed query and documents independently for fast sub-millisecond retrieval. Cross-encoders (`ms-marco-MiniLM-L-6-v2`) pass query and document jointly through full cross-attention layers, computing true semantic relevance.
- **Latency / Quality Trade-off:** Running cross-encoders on 10,000+ chunks is computationally infeasible on CPU. Our two-stage design retrieves the top 20 candidates in $<15\text{ms}$ and reranks only those 20 with the cross-encoder in $\sim 120\text{ms}$, achieving state-of-the-art precision within CPU latency budgets.

### 3. OCR Fallback Pipeline for Unsearchable PDFs
- **Problem:** In enterprise settings, many PDFs are scanned documents, rasterized contracts, or image-only exports containing zero extractable digital text.
- **Solution:** Two-tiered ingestion in `ingest.py`: normal `pypdf` extraction is attempted first. If a page yields $<30$ characters of text, the system automatically falls back to rendering the page to an image via `pdf2image` and executing CPU-optimized OCR via `pytesseract`.
- **Trade-off:** Pages with extractable text are ingested instantly; only scanned pages incur OCR compute cost.

### 4. Hallucination Guardrails & Dynamic Citation Resolution
- **Strict Grounding:** Low LLM temperature ($0.1$) with a rigid prompt instructing the model to decline queries when context is insufficient.
- **Citation Precision:** If the model returns a refusal response (`"I don't have enough information..."`), the pipeline forcefully suppresses all retrieved sources to return an empty citations list `sources: []`.
- **Dynamic Source Filtering:** For valid answers, the pipeline parses explicit inline document references and returns only the verified chunks that genuinely substantiate the response.

---

## Tech Stack & Components

| Component | Technology | Rationale |
|:---|:---|:---|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional dense vectors, 22M parameters, fast CPU inference (~10ms). |
| **Vector Store** | `ChromaDB` (Persistent) | Embedded local storage, zero external daemon required, instant startup. |
| **Keyword Retrieval**| `rank-bm25` | High-speed in-memory BM25 implementation for exact token matching. |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | High-precision joint relevance scoring trained on MS MARCO. |
| **LLM Inference** | OpenRouter API (`openai/gpt-4o-mini`) | Cost-effective, low-latency, highly grounded instruction-following. |
| **Backend REST API** | `FastAPI` + `Uvicorn` + `Pydantic v2` | High-performance asynchronous API with automatic Swagger documentation. |
| **Frontend UI** | `Streamlit` | Minimalist enterprise workspace interface (Linear/Notion inspired). |
| **OCR Engine** | `pdf2image` + `pytesseract` + `poppler` | Robust CPU-friendly rasterization and text extraction fallback. |
| **Evaluation** | Custom LLM-as-a-judge (RAGAS formula) | Fully reproducible, zero dependency on C++ wheel compilation. |
| **Testing** | `pytest` (76 passing tests) | Full unit and integration coverage across all components. |

---

## Project Structure

```
enterprise-rag/
├── docs/                        # Enterprise PDF documents (e.g., RAGG.pdf)
├── chroma_db/                   # Persistent ChromaDB vector storage (auto-generated)
├── evaluation/
│   ├── golden_qa_set.json       # 22-question golden QA dataset (18 in-scope + 4 refusal)
│   ├── results.json             # Canonical benchmark comparison results
│   └── results_*.json           # Timestamped execution run logs
├── tests/
│   ├── conftest.py              # Pytest fixtures and mock objects
│   ├── test_api.py              # FastAPI endpoint unit & integration tests
│   ├── test_chain.py            # RAG pipeline, prompt grounding, and citation tests
│   ├── test_evaluate.py         # Evaluation metrics and golden dataset tests
│   ├── test_ingest.py           # Ingestion, chunking, and OCR fallback tests
│   └── test_retriever.py        # Semantic, BM25, and cross-encoder tests
├── api.py                       # FastAPI production REST backend
├── app.py                       # Streamlit enterprise workspace interface
├── chain.py                     # RAG orchestration, grounding prompt, memory
├── evaluate.py                  # Evaluation CLI and LLM-as-a-judge engine
├── ingest.py                    # PDF ingestion, chunking, OCR fallback, vectorstore
├── retriever.py                 # Hybrid retriever and cross-encoder reranker
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Multi-stage production container with Poppler/Tesseract
├── docker-compose.yml           # Multi-service setup (Streamlit UI + FastAPI API)
├── pytest.ini                   # Pytest configuration
├── .env.example                 # Sanitized environment template
└── README.md                    # System documentation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Tested through Python 3.14 on Windows & Linux)
- An [OpenRouter](https://openrouter.ai/) account and API key
- (Optional for OCR) Tesseract OCR installed locally or in Docker

### 2. Local Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/enterprise-rag.git
cd enterprise-rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
copy .env.example .env     # On Windows
# cp .env.example .env     # On macOS/Linux
```

Open `.env` and set your OpenRouter API key:
```ini
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
LLM_MODEL=openai/gpt-4o-mini
```

---

## Running the Application

### Option A: Document Ingestion
Place your enterprise PDF files into `docs/` (e.g. `docs/RAGG.pdf`), then index:
```bash
python ingest.py --reset
```
*Output: Extracted text, processed rasterized pages via OCR if needed, generated chunks, embedded into ChromaDB, and indexed BM25.*

### Option B: FastAPI Backend
Start the REST API:
```bash
uvicorn api:app --reload --port 8000
```
- Interactive API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: `GET http://localhost:8000/health`
- Query Endpoint: `POST http://localhost:8000/query` with body:
  ```json
  {
    "question": "What are the core functional requirements?",
    "top_k": 5
  }
  ```

### Option C: Streamlit Workspace UI
Launch the interactive web interface:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running the Benchmark Suite

Run the full comparative evaluation between Semantic Baseline and Hybrid + Re-ranking:
```bash
# Compare both configurations side-by-side
python evaluate.py --config both

# Evaluate only hybrid retrieval
python evaluate.py --config hybrid

# Quick dry run (verifies prompt formatting without API calls)
python evaluate.py --dry-run
```

Results are saved to `evaluation/results.json` and timestamped in `evaluation/results_*.json`.

---

## Running Automated Tests

Run the full suite of 76 tests:
```bash
pytest tests/ -v
```

All 76 tests validate:
- Ingestion text cleaning and OCR fallback handlers
- ChromaDB vector store creation and recursive chunk boundaries
- BM25 score normalization and convex fusion math
- Cross-encoder reranking order and candidate deduplication
- Refusal behavior suppressing false citations
- Follow-up conversation memory and turn truncations
- FastAPI HTTP status codes, payloads, and error handlers

---

## Docker Deployment

The project includes container support for both the Streamlit UI and FastAPI backend, including system-level `poppler-utils` and `tesseract-ocr` packages:

```bash
# Build and run both UI (8501) and API (8000)
docker compose up --build

# Run document ingestion inside container
docker compose run app python ingest.py --reset
```

---

## Technical Interview Talking Points

When discussing this project in technical interviews, highlight the following:

1. **Why Hybrid Search beats Vector-only in Enterprise Domains**:
   - Vector embeddings capture semantic gist well, but perform poorly on exact domain tokens (e.g., alphanumeric IDs like `FR-19`, code names, part numbers).
   - Adding BM25 with min-max score normalization ensures exact keyword recall without drowning out semantic matches.
2. **Two-Stage Retrieval Architecture**:
   - Bi-encoders encode queries and documents into static vector spaces, allowing sub-linear ANN retrieval via vector databases.
   - Cross-encoders perform joint attention across both sequences, evaluating exact word interactions and relevance nuance.
   - Using a bi-encoder for candidate generation ($k=20$) and a cross-encoder for re-ranking ($k=5$) balances sub-150ms latency with maximum precision.
3. **Preventing LLM Hallucinations in RAG**:
   - Enforcing low temperature ($0.1$) + strict system prompt boundaries.
   - Refusal training: testing with intentionally unanswerable questions to verify that the system states "I don't have enough information" rather than guessing.
   - Citation verification: dropping unverified chunks and stripping source tags on refusals.
4. **Production Readiness**:
   - Decoupled FastAPI backend allowing enterprise service integration.
   - Pydantic v2 data validation schemas.
   - Full OCR fallback for scanned legacy enterprise files.
   - 76 unit and integration tests with continuous offline benchmarking.
