"""
DocIQ — Evaluation Engine
Metrics: ROUGE scores, retrieval quality, latency tracking, confidence calibration.

For a production system, you would add:
  - BERTScore (semantic similarity vs reference)
  - Faithfulness scoring (Ragas framework)
  - Answer relevancy (Ragas)
  - Context recall and precision (Ragas)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Per-query performance record."""
    query: str
    mode: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    top_similarity: float
    avg_confidence: float
    num_chunks_retrieved: int
    timestamp: float = field(default_factory=time.time)
    rouge_scores: Optional[Dict] = None
    answer_length: int = 0


class EvaluationEngine:
    """
    Tracks and computes quality metrics for DocIQ pipeline.
    """

    def __init__(self):
        self.query_history: List[QueryMetrics] = []
        self._rouge_scorer = None

    def _get_rouge_scorer(self):
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"],
                    use_stemmer=True,
                )
            except ImportError:
                logger.warning("rouge-score not installed. ROUGE metrics disabled.")
        return self._rouge_scorer

    def record_query(self, response, query: str, mode: str = "qa") -> QueryMetrics:
        """Record metrics for a completed RAG response."""
        metrics = QueryMetrics(
            query=query,
            mode=mode,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=response.total_time_ms,
            top_similarity=response.top_similarity,
            avg_confidence=response.avg_confidence,
            num_chunks_retrieved=len(response.retrieved_chunks),
            answer_length=len(response.answer),
        )
        self.query_history.append(metrics)
        return metrics

    def compute_rouge(self, prediction: str, reference: str) -> Dict:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L against a reference answer."""
        scorer = self._get_rouge_scorer()
        if scorer is None:
            return {"error": "rouge-score not installed"}

        scores = scorer.score(reference, prediction)
        return {
            "rouge1": {
                "precision": round(scores["rouge1"].precision, 4),
                "recall": round(scores["rouge1"].recall, 4),
                "f1": round(scores["rouge1"].fmeasure, 4),
            },
            "rouge2": {
                "precision": round(scores["rouge2"].precision, 4),
                "recall": round(scores["rouge2"].recall, 4),
                "f1": round(scores["rouge2"].fmeasure, 4),
            },
            "rougeL": {
                "precision": round(scores["rougeL"].precision, 4),
                "recall": round(scores["rougeL"].recall, 4),
                "f1": round(scores["rougeL"].fmeasure, 4),
            },
        }

    def get_aggregate_stats(self) -> Dict:
        """Compute aggregate performance statistics across all queries."""
        if not self.query_history:
            return {"message": "No queries recorded yet"}

        def avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else 0

        retrieval_times = [q.retrieval_time_ms for q in self.query_history]
        generation_times = [q.generation_time_ms for q in self.query_history]
        total_times = [q.total_time_ms for q in self.query_history]
        similarities = [q.top_similarity for q in self.query_history]
        confidences = [q.avg_confidence for q in self.query_history]

        return {
            "total_queries": len(self.query_history),
            "latency": {
                "avg_retrieval_ms": avg(retrieval_times),
                "avg_generation_ms": avg(generation_times),
                "avg_total_ms": avg(total_times),
                "p95_total_ms": round(sorted(total_times)[int(len(total_times) * 0.95)], 2)
                                if len(total_times) >= 10 else "N/A (need 10+ queries)",
            },
            "retrieval_quality": {
                "avg_top_similarity": avg(similarities),
                "avg_confidence": avg(confidences),
                "avg_chunks_retrieved": avg([q.num_chunks_retrieved for q in self.query_history]),
            },
            "query_modes": {
                mode: sum(1 for q in self.query_history if q.mode == mode)
                for mode in set(q.mode for q in self.query_history)
            },
        }

    def get_per_query_table(self) -> List[Dict]:
        """Returns query history as a list of dicts for display."""
        return [
            {
                "Query": q.query[:60] + "..." if len(q.query) > 60 else q.query,
                "Mode": q.mode,
                "Total (ms)": round(q.total_time_ms),
                "Retrieval (ms)": round(q.retrieval_time_ms),
                "Generation (ms)": round(q.generation_time_ms),
                "Top Similarity": round(q.top_similarity, 3),
                "Confidence": f"{q.avg_confidence:.0%}",
                "Chunks": q.num_chunks_retrieved,
            }
            for q in self.query_history
        ]

    def retrieval_score_analysis(self, results) -> Dict:
        """Analyze retrieval result quality for a single query."""
        if not results:
            return {"quality": "No results", "recommendation": "Lower similarity threshold"}

        scores = [r.similarity_score for r in results]
        top = scores[0]

        quality = "Excellent" if top > 0.75 else \
                  "Good" if top > 0.55 else \
                  "Moderate" if top > 0.35 else "Low"

        score_drop = (scores[0] - scores[-1]) if len(scores) > 1 else 0

        return {
            "quality": quality,
            "top_score": round(top, 4),
            "score_spread": round(score_drop, 4),
            "result_count": len(results),
            "sections_covered": list({r.chunk.section_title for r in results}),
            "recommendation": (
                "Results are highly relevant" if quality == "Excellent" else
                "Results are reasonably relevant" if quality == "Good" else
                "Consider rephrasing for better results" if quality == "Moderate" else
                "Query may not match document content well"
            ),
        }
