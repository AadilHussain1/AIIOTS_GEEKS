"""
DocIQ — Embedding Engine
Generates dense vector embeddings using sentence-transformers.

Model Selection:
  - all-MiniLM-L6-v2:      384-dim, 80MB, fast (default)
  - BAAI/bge-large-en-v1.5: 1024-dim, 1.3GB, best quality
  - all-mpnet-base-v2:      768-dim, balanced quality/speed

Optimization:
  - Batch processing for throughput
  - Model caching (loaded once, reused)
  - Optional GPU acceleration via CUDA
  - Normalize embeddings for cosine similarity efficiency
"""

import logging
import time
from typing import List, Optional
import numpy as np

from core.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Wrapper around sentence-transformers for chunk and query embedding.

    Architecture Decision:
      We use sentence-transformers over OpenAI embeddings because:
      1. No API cost for embedding (critical for large docs)
      2. Runs fully offline
      3. all-MiniLM-L6-v2 offers excellent quality-to-speed ratio
      4. Embeddings are normalized by default (cosine ≡ dot product)
    """

    _instance = None  # singleton embedding model

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        """Load the sentence-transformer model (cached after first load)."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            t0 = time.time()
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded: dim={self.embedding_dim}, "
                        f"time={time.time()-t0:.1f}s")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of document chunks.
        Returns: np.ndarray of shape (n_chunks, embedding_dim), float32, normalized.
        """
        if not chunks:
            return np.array([])

        texts = [chunk.text for chunk in chunks]
        return self._embed_texts(texts, batch_size, show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Applies query prefix optimization for asymmetric retrieval models
        (BGE models expect "Represent this sentence: " prefix for queries).
        """
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        result = self._embed_texts([query])
        return result[0]

    def _embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Core embedding function with batching and normalization."""
        t0 = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,    # L2 normalize → cosine = dot product
            convert_to_numpy=True,
        )

        elapsed = time.time() - t0
        logger.debug(f"Embedded {len(texts)} texts in {elapsed:.2f}s "
                     f"({len(texts)/elapsed:.0f} texts/sec)")
        return embeddings.astype(np.float32)

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
        }
