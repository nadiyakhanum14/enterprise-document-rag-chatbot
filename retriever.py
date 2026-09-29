"""
retriever.py — Hybrid retrieval with cross-encoder re-ranking.

Architecture:
  User Query
      │
      ├─── Semantic Search (ChromaDB cosine similarity)
      │        Returns top FETCH_K candidates with similarity scores [0, 1]
      │
      ├─── BM25 Keyword Search (rank-bm25)
      │        Returns top FETCH_K candidates with BM25 scores (raw, unbounded)
      │
      ├─── Score Normalization + Fusion
      │        Both score vectors normalized to [0, 1] independently
      │        Combined score = alpha * semantic + (1 - alpha) * bm25
      │        alpha=0.5 gives balanced hybrid; alpha=1.0 is semantic-only
      │
      └─── Cross-Encoder Re-ranking
               Full query-document pairs scored by a bi-directional model
               More expensive but higher precision than bi-encoders
               Only applied to top N candidates (not the full corpus)
               Returns final TOP_K chunks

Design decisions:
  - BM25 catches exact matches (names, codes, section numbers, numbers, acronyms)
    that semantic embeddings may miss because they represent meaning, not tokens
  - Cross-encoder (ms-marco-MiniLM-L-6-v2): ~22M params, fast on CPU,
    trained on MS MARCO passage retrieval — good general relevance scorer
  - Normalization is critical: raw BM25 scores can be 0–50+, cosine scores
    are 0–1. Without normalization, one signal drowns the other.
"""

import os
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ALPHA = float(os.getenv("RETRIEVAL_ALPHA", "0.5"))
FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "20"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a score array to [0, 1].
    If all scores are equal (e.g., all zero), return zeros.
    """
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


class HybridRetriever:
    """
    Two-stage retriever:
      Stage 1 — Hybrid retrieval (semantic + BM25 fusion)
      Stage 2 — Cross-encoder re-ranking
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        fetch_k: int = FETCH_K,
        alpha: float = ALPHA,
    ):
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.alpha = alpha  # weight for semantic score; (1-alpha) for BM25

        self._load_vectorstore()
        self._build_bm25()
        self._load_cross_encoder()

    # ── Initialization ────────────────────────────────────────────────────────

    def _load_vectorstore(self):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self._embeddings,
            collection_name="enterprise_docs",
        )
        # Pull all stored documents once for BM25 index construction
        result = self.vectorstore.get(include=["documents", "metadatas"])
        self._corpus_texts: List[str] = result.get("documents") or []
        self._corpus_metas: List[dict] = result.get("metadatas") or []

    def _build_bm25(self):
        if not self._corpus_texts:
            self.bm25 = None
            return

        tokenized = [text.lower().split() for text in self._corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def _load_cross_encoder(self):
        # ms-marco-MiniLM-L-6-v2: ~22M params, CPU-friendly
        # Trained on MS MARCO: strong at passage-level relevance judgment
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _semantic_search(self, query: str) -> Tuple[List[Document], np.ndarray]:
        """Return (docs, similarity_scores) from ChromaDB cosine similarity."""
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.fetch_k
        )
        if not results:
            return [], np.array([])
        docs, scores = zip(*results)
        return list(docs), np.array(scores, dtype=float)

    def _bm25_search(self, query: str) -> Tuple[List[Document], np.ndarray]:
        """Return (docs, bm25_scores) for the top fetch_k BM25 results."""
        if self.bm25 is None:
            return [], np.array([])

        raw_scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(raw_scores)[::-1][: self.fetch_k]

        docs, scores = [], []
        for idx in top_indices:
            docs.append(
                Document(
                    page_content=self._corpus_texts[idx],
                    metadata=self._corpus_metas[idx] if idx < len(self._corpus_metas) else {},
                )
            )
            scores.append(raw_scores[idx])

        return docs, np.array(scores, dtype=float)

    def _fuse_results(
        self,
        sem_docs: List[Document],
        sem_scores: np.ndarray,
        bm25_docs: List[Document],
        bm25_scores: np.ndarray,
    ) -> List[Tuple[Document, float]]:
        """
        Merge semantic and BM25 results with normalized score fusion.

        Each document gets:
            fused_score = alpha * norm_semantic + (1 - alpha) * norm_bm25
        Documents that appear in only one list get 0 for the missing score.
        """
        # Normalize scores independently
        norm_sem = _normalize(sem_scores) if len(sem_scores) else np.array([])
        norm_bm25 = _normalize(bm25_scores) if len(bm25_scores) else np.array([])

        # Build score lookup: content_key → (doc, semantic_score, bm25_score)
        score_map: dict = {}

        for doc, score in zip(sem_docs, norm_sem):
            key = doc.page_content[:120]
            score_map[key] = {"doc": doc, "sem": float(score), "bm25": 0.0}

        for doc, score in zip(bm25_docs, norm_bm25):
            key = doc.page_content[:120]
            if key in score_map:
                score_map[key]["bm25"] = float(score)
            else:
                score_map[key] = {"doc": doc, "sem": 0.0, "bm25": float(score)}

        # Compute fused score and sort
        fused = []
        for entry in score_map.values():
            fused_score = self.alpha * entry["sem"] + (1 - self.alpha) * entry["bm25"]
            fused.append((entry["doc"], fused_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def _rerank(self, query: str, candidates: List[Document]) -> List[Document]:
        """
        Cross-encoder re-ranking.
        Score each (query, document) pair jointly — much more precise than
        bi-encoder similarity but O(n) in candidates, so only run on top candidates.
        """
        if not candidates:
            return []

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.top_k]]

    def retrieve(self, query: str) -> List[Document]:
        """
        Full retrieval pipeline: hybrid search → score fusion → cross-encoder rerank.
        Returns top_k most relevant Document objects with source metadata.
        """
        sem_docs, sem_scores = self._semantic_search(query)
        bm25_docs, bm25_scores = self._bm25_search(query)

        fused = self._fuse_results(sem_docs, sem_scores, bm25_docs, bm25_scores)
        candidates = [doc for doc, _ in fused[: self.fetch_k]]

        return self._rerank(query, candidates)

    def retrieve_semantic_only(self, query: str) -> List[Document]:
        """
        Semantic-only retrieval (no BM25, no re-ranking).
        Used for baseline comparison in evaluation.
        """
        docs, _ = self._semantic_search(query)
        return docs[: self.top_k]

    def get_corpus_size(self) -> int:
        """Return number of indexed chunks."""
        return len(self._corpus_texts)
