"""
DocIQ — Session Manager & Utilities
"""

import logging
import traceback
import sys
import streamlit as st

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def initialize_session_state():
    defaults = {
        "parsed_doc": None,
        "chunks": [],
        "doc_stats": {},
        "chunking_stats": {},
        "embedding_engine": None,
        "vector_store": None,
        "llm_engine": None,
        "rag_pipeline": None,
        "evaluator": None,
        "messages": [],
        "last_response": None,
        "current_summary": "",
        "current_extraction": {},
        "processing": False,
        "doc_loaded": False,
        "active_tab": "chat",
        "selected_section": None,
        "api_key_set": False,
        "_last_build_error": "",
        "_last_file_key": "",
        "top_k": 5,
        "use_mmr": True,
        "similarity_threshold": 0.20,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "llm_backend": "ollama",
        "anthropic_api_key": "",
        "openai_api_key": "",
        "ollama_model": "mistral:7b",
        "ollama_base_url": "http://localhost:11434",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


@st.cache_resource(show_spinner=False)
def load_embedding_engine(model_name: str = "all-MiniLM-L6-v2"):
    from core.embeddings import EmbeddingEngine
    return EmbeddingEngine(model_name=model_name)


def build_pipeline_from_state() -> bool:
    """
    Build RAG pipeline. LLM is NOT tested here — it is only
    called when the user sends a message. This avoids Ollama
    cold-start timeouts during the index build phase.
    """
    st.session_state["_last_build_error"] = ""
    try:
        from core.chunker import DocumentChunker
        from core.vector_store import VectorStore
        from core.llm_engine import LLMEngine, AnthropicProvider, OpenAIProvider, OllamaProvider
        from core.rag_pipeline import RAGPipeline, ConversationMemory
        from core.evaluator import EvaluationEngine

        doc = st.session_state.parsed_doc
        if doc is None:
            st.session_state["_last_build_error"] = "No document loaded. Upload a file first."
            return False

        # ── Step 1: Chunk ──
        chunker = DocumentChunker(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
        chunks = chunker.chunk_document(doc)
        if not chunks:
            st.session_state["_last_build_error"] = "Chunking produced 0 chunks. Document may be empty."
            return False
        st.session_state.chunks = chunks
        st.session_state.chunking_stats = chunker.get_chunking_stats(chunks)

        # ── Step 2: Embed ──
        embed_engine = load_embedding_engine()
        st.session_state.embedding_engine = embed_engine
        embeddings = embed_engine.embed_chunks(chunks, show_progress=False)

        # ── Step 3: FAISS index ──
        vs = VectorStore()
        vs.build_index(chunks, embeddings)
        st.session_state.vector_store = vs

        # ── Step 4: LLM provider (NO network call — lazy init) ──
        backend = st.session_state.llm_backend

        if backend == "anthropic":
            key = st.session_state.anthropic_api_key.strip()
            if not key:
                st.session_state["_last_build_error"] = (
                    "Anthropic API key is empty.\n"
                    "Paste it in the sidebar or switch the backend to Ollama."
                )
                return False
            provider = AnthropicProvider(api_key=key)

        elif backend == "openai":
            key = st.session_state.openai_api_key.strip()
            if not key:
                st.session_state["_last_build_error"] = (
                    "OpenAI API key is empty.\n"
                    "Paste it in the sidebar or switch the backend to Ollama."
                )
                return False
            provider = OpenAIProvider(api_key=key)

        elif backend == "ollama":
            # Just create the provider — no network call yet.
            # The first actual call happens when the user chats.
            model = st.session_state.ollama_model.strip() or "mistral:7b"
            base_url = st.session_state.ollama_base_url.strip().rstrip("/")
            provider = OllamaProvider(model=model, base_url=base_url)

        else:
            st.session_state["_last_build_error"] = f"Unknown backend: {backend}"
            return False

        llm = LLMEngine(provider)
        st.session_state.llm_engine = llm

        # ── Step 5: Wire up pipeline ──
        pipeline = RAGPipeline(
            embedding_engine=embed_engine,
            vector_store=vs,
            llm_engine=llm,
            memory=ConversationMemory(window_size=10),
        )
        st.session_state.rag_pipeline = pipeline
        st.session_state.evaluator = EvaluationEngine()
        st.session_state.doc_loaded = True
        st.session_state.messages = []
        return True

    except Exception as e:
        full_tb = traceback.format_exc()
        logger.error(f"Pipeline build failed:\n{full_tb}")
        st.session_state["_last_build_error"] = f"{type(e).__name__}: {e}\n\n{full_tb}"
        return False


def format_number(n: int) -> str:
    """Display large numbers compactly: 1234 -> 1.2K, 1200000 -> 1.2M"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 10_000:
        return f"{n/1_000:.0f}K"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)
