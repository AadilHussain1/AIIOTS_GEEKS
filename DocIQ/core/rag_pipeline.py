"""
DocIQ — RAG Pipeline
Orchestrates the complete Retrieval-Augmented Generation pipeline.

Data Flow:
  Query
    → Memory injection (conversation context)
    → Query embedding
    → FAISS retrieval (MMR)
    → Context assembly (token-budget aware)
    → Prompt construction
    → LLM generation
    → Response + metadata
    → Memory update
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Generator

from core.embeddings import EmbeddingEngine
from core.vector_store import VectorStore, RetrievalResult
from core.llm_engine import LLMEngine
from core.chunker import DocumentChunk
from prompts.templates import (
    build_rag_qa_prompt,
    build_summary_prompt,
    build_extraction_prompt,
    build_hierarchical_summary_prompt,
    build_memory_compression_prompt,
    build_conversation_messages,
    RAG_QA_SYSTEM,
)
from core.document_processor import ParsedDocument
from config import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# RESPONSE MODEL
# ──────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: List[RetrievalResult]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    top_similarity: float
    avg_confidence: float
    sources: List[Dict] = field(default_factory=list)
    mode: str = "qa"

    def __post_init__(self):
        self.sources = [
            {
                "section": r.chunk.section_title,
                "similarity": f"{r.similarity_score:.3f}",
                "confidence": r.confidence_label,
                "rank": r.rank + 1,
            }
            for r in self.retrieved_chunks
        ]


# ──────────────────────────────────────────────
# CONVERSATION MEMORY
# ──────────────────────────────────────────────

class ConversationMemory:
    """
    Sliding window + compression memory for multi-turn conversation.

    Strategy:
      - Keep last N turns verbatim (recent memory)
      - When buffer exceeds threshold, compress older turns via LLM
      - Inject compressed summary + recent turns into each new prompt
    """

    def __init__(self, window_size: int = 10, compression_threshold: int = 20):
        self.window_size = window_size
        self.compression_threshold = compression_threshold
        self.turns: List[Dict] = []          # {"role": ..., "content": ...}
        self.compressed_summary: str = ""     # LLM-compressed older history

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def get_recent_turns(self) -> List[Dict]:
        """Returns last N turns for inclusion in LLM messages."""
        return self.turns[-self.window_size * 2:]  # *2 because each turn = user+assistant

    def get_history_string(self) -> str:
        """Plain-text history for prompt injection."""
        recent = self.get_recent_turns()
        if not recent:
            return ""
        lines = []
        for turn in recent:
            role = "User" if turn["role"] == "user" else "Assistant"
            # Truncate long turns in the history string
            content = turn["content"][:500] + "..." if len(turn["content"]) > 500 else turn["content"]
            lines.append(f"{role}: {content}")
        history = "\n".join(lines)
        if self.compressed_summary:
            return f"[Earlier conversation summary: {self.compressed_summary}]\n\n{history}"
        return history

    def should_compress(self) -> bool:
        return len(self.turns) >= self.compression_threshold

    async def compress(self, llm: LLMEngine):
        """Compress older memory to free up context window space."""
        history_str = "\n".join(
            f"{'User' if t['role']=='user' else 'Assistant'}: {t['content'][:300]}"
            for t in self.turns[:-self.window_size]
        )
        system, user_msg = build_memory_compression_prompt(history_str)
        self.compressed_summary = llm.generate(system, user_msg, max_tokens=300)
        # Keep only recent window
        self.turns = self.turns[-self.window_size:]
        logger.info("Compressed conversation memory")

    def clear(self):
        self.turns = []
        self.compressed_summary = ""

    def to_display(self) -> List[Dict]:
        """Returns full turn history for UI display."""
        return self.turns


# ──────────────────────────────────────────────
# RAG PIPELINE
# ──────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG orchestrator.

    Responsibilities:
      - Route queries to QA / summarization / extraction
      - Manage retrieval with configurable parameters
      - Assemble token-budget-aware context windows
      - Handle streaming and non-streaming generation
      - Track per-query performance metrics
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        llm_engine: LLMEngine,
        memory: Optional[ConversationMemory] = None,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.llm_engine = llm_engine
        self.memory = memory or ConversationMemory(
            window_size=CONFIG.memory_window
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        use_mmr: bool = True,
        similarity_threshold: float = 0.20,
        section_filter: Optional[str] = None,
        max_context_tokens: int = 5000,
    ) -> RAGResponse:
        """
        Main RAG QA method.
        Returns full answer with retrieval metadata.
        """
        t_total = time.time()

        # ── Step 1: Embed query ──
        query_embedding = self.embedding_engine.embed_query(question)

        # ── Step 2: Retrieve relevant chunks ──
        t_ret = time.time()
        results = self.vector_store.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            use_mmr=use_mmr,
            section_filter=section_filter,
        )
        retrieval_time = (time.time() - t_ret) * 1000

        if not results:
            return RAGResponse(
                answer="I could not find relevant information in the document to answer this question. "
                       "Please try rephrasing your question or ask about a different topic covered in the document.",
                retrieved_chunks=[],
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0,
                total_time_ms=(time.time() - t_total) * 1000,
                top_similarity=0.0,
                avg_confidence=0.0,
            )

        # ── Step 3: Build context (token budget management) ──
        budget_results = self._apply_token_budget(results, max_context_tokens)

        # ── Step 4: Build prompt ──
        history_str = self.memory.get_history_string()
        system_prompt, user_message = build_rag_qa_prompt(
            question=question,
            retrieved_chunks=budget_results,
            conversation_history=history_str,
        )

        # ── Step 5: Generate answer ──
        t_gen = time.time()
        answer = self.llm_engine.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=CONFIG.model.max_response_tokens,
            temperature=0.1,
        )
        generation_time = (time.time() - t_gen) * 1000

        # ── Step 6: Update memory ──
        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", answer)

        # ── Step 7: Compute metrics ──
        top_sim = results[0].similarity_score if results else 0.0
        avg_conf = sum(r.confidence for r in budget_results) / max(1, len(budget_results))

        return RAGResponse(
            answer=answer,
            retrieved_chunks=budget_results,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=(time.time() - t_total) * 1000,
            top_similarity=top_sim,
            avg_confidence=avg_conf,
            mode="qa",
        )

    def answer_stream(
        self,
        question: str,
        top_k: int = 5,
        use_mmr: bool = True,
        similarity_threshold: float = 0.20,
        section_filter: Optional[str] = None,
    ) -> Generator:
        """
        Streaming RAG QA.
        Yields: ("chunk", text_token) | ("metadata", RAGResponse)
        """
        t_total = time.time()

        # Embed + retrieve (non-streamed)
        query_embedding = self.embedding_engine.embed_query(question)
        results = self.vector_store.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            use_mmr=use_mmr,
            section_filter=section_filter,
        )
        retrieval_time = (time.time() - t_total) * 1000

        if not results:
            yield ("chunk", "I could not find relevant information in the document to answer this question.")
            return

        history_str = self.memory.get_history_string()
        system_prompt, user_message = build_rag_qa_prompt(
            question=question,
            retrieved_chunks=results,
            conversation_history=history_str,
        )

        # Build message history for streaming
        recent_turns = self.memory.get_recent_turns()
        messages = recent_turns + [{"role": "user", "content": user_message}]

        # Stream response
        full_answer = ""
        t_gen = time.time()
        for token in self.llm_engine.generate_stream(
            system_prompt=system_prompt,
            messages=messages,
        ):
            full_answer += token
            yield ("chunk", token)

        generation_time = (time.time() - t_gen) * 1000

        # Update memory
        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", full_answer)

        # Yield metadata
        top_sim = results[0].similarity_score if results else 0.0
        avg_conf = sum(r.confidence for r in results) / max(1, len(results))
        yield ("metadata", RAGResponse(
            answer=full_answer,
            retrieved_chunks=results,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=(time.time() - t_total) * 1000,
            top_similarity=top_sim,
            avg_confidence=avg_conf,
        ))

    def summarize(
        self,
        doc: ParsedDocument,
        mode: str = "executive",
    ) -> str:
        """
        Multi-mode document summarization.
        Modes: tldr | executive | technical | bullets | hierarchical
        """
        if mode == "hierarchical":
            return self._hierarchical_summarize(doc)

        system_prompt, user_message = build_summary_prompt(
            mode=mode,
            document_text=doc.raw_text,
            doc_filename=doc.filename,
        )
        return self.llm_engine.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=1500,
            temperature=0.2,
        )

    def _hierarchical_summarize(self, doc: ParsedDocument) -> str:
        """Section-by-section then global synthesis summarization."""
        sections = [
            (s.title, s.content)
            for s in doc.sections
            if s.word_count > 20
        ][:12]  # Limit to 12 sections to stay within context

        system_prompt, user_message = build_hierarchical_summary_prompt(
            sections=sections,
            doc_filename=doc.filename,
        )
        return self.llm_engine.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=2000,
            temperature=0.2,
        )

    def extract(
        self,
        doc: ParsedDocument,
        custom_schema: dict = None,
    ) -> dict:
        """
        Structured JSON information extraction.
        Returns parsed dict or raw string if JSON parsing fails.
        """
        system_prompt, user_message = build_extraction_prompt(
            document_text=doc.raw_text,
            extraction_schema=custom_schema,
        )
        raw = self.llm_engine.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=2000,
            temperature=0.0,  # Zero temp for deterministic structured output
        )

        # Strip markdown fences if present
        clean = raw.strip()
        for fence in ["```json", "```JSON", "```"]:
            if clean.startswith(fence):
                clean = clean[len(fence):].strip()
            if clean.endswith("```"):
                clean = clean[:-3].strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Returning raw response.")
            return {"raw_extraction": raw, "parse_error": str(e)}

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Pure semantic search — returns retrieved chunks without generation.
        Useful for exploring document content.
        """
        query_embedding = self.embedding_engine.embed_query(query)
        return self.vector_store.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            use_mmr=False,
            section_filter=section_filter,
        )

    def _apply_token_budget(
        self,
        results: List[RetrievalResult],
        max_tokens: int,
    ) -> List[RetrievalResult]:
        """
        Token budget management: include as many chunks as fit within the
        context window budget, in order of relevance.
        """
        from core.chunker import count_tokens
        selected = []
        used_tokens = 0

        for result in results:
            chunk_tokens = result.chunk.token_count
            if used_tokens + chunk_tokens <= max_tokens:
                selected.append(result)
                used_tokens += chunk_tokens
            else:
                break  # Stop adding if budget exceeded

        if not selected and results:
            # At minimum include the top result (truncated)
            selected = [results[0]]

        return selected

    def reset_memory(self):
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
