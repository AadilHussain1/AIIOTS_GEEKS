"""
DocIQ — Vector Store
FAISS-backed semantic retrieval with MMR and confidence scoring.

Architecture:
  - Index Type: IndexFlatIP (inner product = cosine on normalized vectors)
  - For large corpora (>100k chunks): switch to IndexIVFFlat with nlist=256
  - Retrieval: k-NN → Optional MMR reranking → Confidence scoring
  - Persistence: Save/load index to disk for session resumption
"""

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.chunker import DocumentChunk

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# RETRIEVAL RESULT
# ──────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    similarity_score: float       # cosine similarity [0,1]
    confidence: float             # normalized confidence score [0,1]
    rank: int                     # retrieval rank (0 = most relevant)
    retrieval_method: str         # "cosine" | "mmr"

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.80:
            return "High"
        elif self.confidence >= 0.55:
            return "Medium"
        else:
            return "Low"


# ──────────────────────────────────────────────
# FAISS VECTOR STORE
# ──────────────────────────────────────────────

class VectorStore:
    """
    FAISS vector store for semantic chunk retrieval.

    Retrieval Pipeline:
      1. Embed query via EmbeddingEngine
      2. FAISS k-NN search (IndexFlatIP, exact search)
      3. Filter by similarity threshold
      4. Optional MMR reranking for diversity
      5. Confidence score normalization
      6. Return RetrievalResult list with metadata
    """

    def __init__(self):
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dim: Optional[int] = None
        self.doc_count: int = 0
        self._faiss = None
        self._load_faiss()

    def _load_faiss(self):
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    def build_index(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Build FAISS index from chunk embeddings.
        Called once after document processing.
        """
        assert len(chunks) == embeddings.shape[0], \
            f"Chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) must match"

        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedding_dim = embeddings.shape[1]
        self.doc_count += 1

        # IndexFlatIP: exact search on inner product (= cosine for normalized vectors)
        # For >50k chunks: use IndexIVFFlat for faster approximate search
        if len(chunks) > 50_000:
            quantizer = self._faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(256, len(chunks) // 10)
            self.index = self._faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist, self._faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = 32
        else:
            self.index = self._faiss.IndexFlatIP(self.embedding_dim)

        self.index.add(self.embeddings)
        logger.info(f"Built FAISS index: {len(chunks)} chunks, dim={self.embedding_dim}")

    def add_chunks(
        self,
        new_chunks: List[DocumentChunk],
        new_embeddings: np.ndarray,
    ) -> None:
        """Incrementally add chunks (for multi-document support)."""
        if self.index is None:
            self.build_index(new_chunks, new_embeddings)
            return

        self.chunks.extend(new_chunks)
        new_emb = new_embeddings.astype(np.float32)
        self.embeddings = np.vstack([self.embeddings, new_emb])
        self.index.add(new_emb)
        self.doc_count += 1
        logger.info(f"Added {len(new_chunks)} chunks. Total: {len(self.chunks)}")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        mmr_top_n: int = 3,
        section_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Main retrieval method.

        Args:
            query_embedding: L2-normalized query vector
            top_k: number of candidates to retrieve before reranking
            similarity_threshold: minimum cosine similarity
            use_mmr: apply Maximal Marginal Relevance reranking
            mmr_lambda: 1.0=pure relevance, 0.0=pure diversity
            mmr_top_n: final number of results after MMR
            section_filter: restrict retrieval to specific section title
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        t0 = time.time()

        # Expand k for pre-MMR filtering
        search_k = min(top_k * 3, len(self.chunks))

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, search_k)
        scores = scores[0]
        indices = indices[0]

        # Filter invalid indices and low similarity
        candidates = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]

            # Section filter
            if section_filter and section_filter not in chunk.section_title:
                continue

            if score >= similarity_threshold:
                candidates.append((chunk, float(score), int(idx)))

        if not candidates:
            logger.warning(f"No chunks above threshold {similarity_threshold}. "
                           f"Best score: {scores[0]:.3f}")
            # Return top-1 with low confidence rather than nothing
            if len(scores) > 0 and indices[0] >= 0:
                chunk = self.chunks[indices[0]]
                return [RetrievalResult(
                    chunk=chunk,
                    similarity_score=float(scores[0]),
                    confidence=float(scores[0]),
                    rank=0,
                    retrieval_method="cosine_fallback",
                )]
            return []

        # Apply MMR for result diversity
        if use_mmr and len(candidates) > mmr_top_n:
            selected = self._mmr(
                query_embedding=query_embedding,
                candidates=candidates,
                top_n=mmr_top_n,
                lmb=mmr_lambda,
            )
            method = "mmr"
        else:
            selected = candidates[:top_k]
            method = "cosine"

        # Compute confidence scores (normalize to [0,1] relative to top score)
        top_score = selected[0][1] if selected else 1.0
        results = []
        for rank, (chunk, score, _) in enumerate(selected):
            confidence = min(1.0, score / max(top_score, 1e-6))
            results.append(RetrievalResult(
                chunk=chunk,
                similarity_score=score,
                confidence=confidence,
                rank=rank,
                retrieval_method=method,
            ))

        elapsed_ms = (time.time() - t0) * 1000
        logger.debug(f"Retrieved {len(results)} chunks in {elapsed_ms:.1f}ms "
                     f"(method={method}, top_score={results[0].similarity_score:.3f})")
        return results

    def _mmr(
        self,
        query_embedding: np.ndarray,
        candidates: List[Tuple],
        top_n: int,
        lmb: float,
    ) -> List[Tuple]:
        """
        Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

        Balances relevance to query and diversity among selected results.
        MMR score = λ * sim(q, d) - (1-λ) * max(sim(d, s)) for s in selected
        """
        selected = []
        remaining = list(candidates)
        candidate_embeddings = {
            idx: self.embeddings[idx] for _, _, idx in candidates
        }

        while len(selected) < top_n and remaining:
            best_score = -np.inf
            best_candidate = None

            for chunk, rel_score, idx in remaining:
                emb = candidate_embeddings[idx]

                if selected:
                    # Similarity to already selected items
                    selected_embs = np.array([
                        candidate_embeddings[s_idx]
                        for _, _, s_idx in selected
                    ])
                    redundancy = np.max(emb @ selected_embs.T)
                else:
                    redundancy = 0.0

                mmr_score = lmb * rel_score - (1 - lmb) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = (chunk, rel_score, idx)

            selected.append(best_candidate)
            remaining = [c for c in remaining if c[2] != best_candidate[2]]

        return selected

    def clear(self) -> None:
        """Reset the vector store (called when a new document is loaded)."""
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.doc_count = 0
        logger.info("Vector store cleared")

    def get_stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "documents_indexed": self.doc_count,
            "embedding_dim": self.embedding_dim,
            "index_type": type(self.index).__name__ if self.index else "None",
            "sections": list({c.section_title for c in self.chunks}),
        }
