"""
DocIQ — Advanced Document Intelligence System
Configuration & Constants
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────
# MODEL CONFIGURATION
# ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    # LLM Backend: "anthropic" | "openai" | "ollama"
    llm_backend: str = "ollama"

    # Anthropic
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # OpenAI (alternative)
    openai_model: str = "gpt-4o"
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Ollama (local open-source alternative)
    ollama_model: str = "llama3.2:3b"        # or mistral, phi3, gemma2
    ollama_base_url: str = "http://localhost:11434"

    # Embedding model (sentence-transformers)
    # Options: "all-MiniLM-L6-v2" (fast), "BAAI/bge-large-en-v1.5" (best quality)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384           # 384 for MiniLM, 1024 for bge-large

    # Context limits
    max_context_tokens: int = 6000
    max_response_tokens: int = 2000


# ──────────────────────────────────────────────
# CHUNKING CONFIGURATION
# ──────────────────────────────────────────────

@dataclass
class ChunkConfig:
    # Token-aware chunking
    chunk_size: int = 512            # tokens per chunk
    chunk_overlap: int = 64          # overlap for context continuity
    min_chunk_size: int = 50         # discard chunks smaller than this

    # Section detection patterns (regex)
    section_patterns: list = field(default_factory=lambda: [
        r"^#{1,6}\s",                # Markdown headers
        r"^\d+\.\s+[A-Z]",          # Numbered sections
        r"^[A-Z][A-Z\s]{3,}$",      # ALL CAPS HEADERS
        r"^(Abstract|Introduction|Conclusion|References|Methodology|Results|Discussion)",
    ])

    # Semantic splitting threshold
    semantic_breakpoint_threshold: float = 0.35


# ──────────────────────────────────────────────
# RETRIEVAL CONFIGURATION
# ──────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    top_k: int = 5                        # chunks to retrieve
    similarity_threshold: float = 0.25    # min cosine similarity
    rerank_top_n: int = 3                 # after reranking
    use_mmr: bool = True                  # Maximal Marginal Relevance
    mmr_lambda: float = 0.7              # 1.0 = pure similarity, 0.0 = pure diversity
    include_section_context: bool = True  # inject section header into chunk


# ──────────────────────────────────────────────
# SUMMARIZATION MODES
# ──────────────────────────────────────────────

SUMMARY_MODES = {
    "tldr":      "TL;DR (2-3 sentences)",
    "executive": "Executive Summary (structured, business-oriented)",
    "technical": "Technical Deep-Dive (preserve methodology & metrics)",
    "bullets":   "Bullet-Point Breakdown (key points only)",
    "hierarchical": "Hierarchical (section-by-section then global)",
}

# ──────────────────────────────────────────────
# SYSTEM-WIDE SETTINGS
# ──────────────────────────────────────────────

@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    app_name: str = "DocIQ"
    app_version: str = "1.0.0"
    app_tagline: str = "Advanced Document Intelligence System"

    max_upload_size_mb: int = 50
    supported_formats: list = field(default_factory=lambda: [".pdf", ".docx", ".txt"])

    # Conversation memory
    memory_window: int = 10           # last N turns to include
    memory_summary_threshold: int = 20  # compress memory after N turns

    # Logging
    log_level: str = "INFO"
    log_file: str = "dociq.log"

    # Evaluation
    enable_evaluation: bool = True
    rouge_variants: list = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])


# Singleton config
CONFIG = AppConfig()
