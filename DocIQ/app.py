"""
╔══════════════════════════════════════════════════════════════╗
║              DocIQ — Document Intelligence System            ║
║          Abstractive RAG · Semantic Search · Extraction      ║
╚══════════════════════════════════════════════════════════════╝

Run: streamlit run app.py
"""

import json
import logging
import time
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="DocIQ — Document Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "DocIQ v1.0 — Advanced Document Intelligence System"},
)

from utils.session import initialize_session_state, build_pipeline_from_state, load_embedding_engine, setup_logging, format_number
from core.document_processor import DocumentProcessor

setup_logging()
logger = logging.getLogger(__name__)
initialize_session_state()


# ════════════════════════════════════════════════
# CUSTOM CSS — Dark Premium Theme
# ════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-deep: #0a0b0f;
    --bg-panel: #111318;
    --bg-card: #161820;
    --bg-hover: #1c1f27;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --accent-dim: rgba(99,102,241,0.12);
    --accent-glow: rgba(99,102,241,0.35);
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-dim: #475569;
    --border: rgba(255,255,255,0.06);
    --border-accent: rgba(99,102,241,0.3);
    --radius: 12px;
    --font-display: 'Syne', sans-serif;
    --font-body: 'Inter', sans-serif;
    --font-mono: 'Space Mono', monospace;
}

/* ── Global Reset ── */
html, body, .stApp {
    background: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
.stApp > div { padding-top: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
    width: 300px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── Custom Upload Area ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-accent) !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 30px var(--accent-glow) !important;
    opacity: 0.92 !important;
}

/* ── Text Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Sliders ── */
.stSlider > div > div > div { color: var(--text-primary) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    gap: 4px !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 0 !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.75rem 1rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-light) !important;
    border-bottom-color: var(--accent-light) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-deep) !important;
    padding: 0 !important;
}

/* ── Code blocks ── */
.stCodeBlock, code { 
    background: var(--bg-panel) !important; 
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* ── Expander ── */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
summary {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.75rem 1rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-panel); }
::-webkit-scrollbar-thumb { background: var(--text-dim); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Custom components ── */
.dociq-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.5rem 1.5rem 1rem;
    border-bottom: 1px solid var(--border);
}
.dociq-logo {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--accent-light);
    letter-spacing: -0.03em;
}
.dociq-tagline {
    font-size: 0.7rem;
    color: var(--text-dim);
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    padding: 0 1.5rem;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    display: block;
}
.doc-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--accent-dim);
    border: 1px solid var(--border-accent);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.75rem 1.25rem;
    font-size: 0.82rem;
    color: var(--accent-light);
    font-family: var(--font-mono);
}
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    margin: 2px;
}
.chat-container {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 80px);
}
.messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
}
.message-user {
    display: flex;
    justify-content: flex-end;
}
.message-user-bubble {
    max-width: 70%;
    background: var(--accent);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.875rem 1.25rem;
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px var(--accent-glow);
}
.message-assistant {
    display: flex;
    justify-content: flex-start;
    gap: 10px;
    align-items: flex-start;
}
.assistant-icon {
    width: 28px;
    height: 28px;
    background: var(--bg-card);
    border: 1px solid var(--border-accent);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}
.message-assistant-bubble {
    max-width: 78%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px 18px 18px 18px;
    padding: 1rem 1.25rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text-primary);
}
.message-meta {
    display: flex;
    gap: 8px;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.meta-tag {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-dim);
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 7px;
}
.meta-tag.good { color: var(--green); border-color: rgba(16,185,129,0.3); }
.meta-tag.med { color: var(--amber); border-color: rgba(245,158,11,0.3); }
.meta-tag.low { color: var(--red); border-color: rgba(239,68,68,0.3); }
.input-bar {
    border-top: 1px solid var(--border);
    padding: 1rem 2rem;
    background: var(--bg-panel);
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    margin: 2px;
    cursor: default;
}
.source-chip:hover {
    border-color: var(--accent);
    color: var(--accent-light);
}
.welcome-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 1.5rem;
    padding: 3rem;
    text-align: center;
    opacity: 0.6;
}
.welcome-icon {
    font-size: 3.5rem;
    filter: grayscale(0.5);
}
.welcome-title {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.welcome-sub {
    color: var(--text-secondary);
    font-size: 0.9rem;
    max-width: 400px;
    line-height: 1.6;
}
.pill-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    max-width: 500px;
}
.example-pill {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
}
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s ease;
}
.section-card:hover { border-color: var(--border-accent); }
.confidence-bar {
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    overflow: hidden;
    margin-top: 6px;
}
.confidence-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════

with st.sidebar:
    # Header
    st.markdown("""
    <div class="dociq-header">
        <span style="font-size:1.6rem">⬡</span>
        <div>
            <div class="dociq-logo">DocIQ</div>
            <div class="dociq-tagline">Document Intelligence System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # LLM Backend selector
    st.markdown('<span class="section-label">🤖 LLM Backend</span>', unsafe_allow_html=True)
    backend = st.selectbox(
        "Backend",
        ["ollama", "anthropic", "openai"],
        index=["ollama", "anthropic", "openai"].index(st.session_state.llm_backend),
        format_func=lambda x: {"ollama": "🦙 Ollama (Free, Local)", "anthropic": "⚡ Anthropic Claude", "openai": "🟢 OpenAI GPT-4o"}[x],
        label_visibility="collapsed",
    )
    st.session_state.llm_backend = backend

    # Show relevant config fields per backend
    if backend == "ollama":
        st.session_state.api_key_set = True   # no key needed
        col_m, col_u = st.columns([3, 2])
        with col_m:
            model = st.text_input("Model", value=st.session_state.ollama_model,
                                  placeholder="mistral:7b", label_visibility="collapsed")
            if model:
                st.session_state.ollama_model = model
        with col_u:
            url = st.text_input("URL", value=st.session_state.ollama_base_url,
                                label_visibility="collapsed")
            if url:
                st.session_state.ollama_base_url = url
        st.caption("Make sure `ollama serve` is running")

    elif backend == "anthropic":
        api_key = st.text_input(
            "API Key", type="password",
            value=st.session_state.anthropic_api_key,
            placeholder="sk-ant-...",
            label_visibility="collapsed",
        )
        if api_key:
            st.session_state.anthropic_api_key = api_key
            st.session_state.api_key_set = True
        st.caption("console.anthropic.com → API Keys")

    elif backend == "openai":
        api_key = st.text_input(
            "API Key", type="password",
            value=st.session_state.openai_api_key,
            placeholder="sk-...",
            label_visibility="collapsed",
        )
        if api_key:
            st.session_state.openai_api_key = api_key
            st.session_state.api_key_set = True
        st.caption("platform.openai.com → API Keys")

    # Document upload
    st.markdown('<span class="section-label">📎 Document</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed",
        help="Supported: PDF, DOCX, TXT, MD",
    )

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_last_file_key") != file_key:
            st.session_state["_last_file_key"] = file_key
            with st.spinner("Parsing document..."):
                try:
                    processor = DocumentProcessor()
                    doc = processor.process(uploaded_file.read(), uploaded_file.name)
                    st.session_state.parsed_doc = doc
                    st.session_state.doc_stats = processor.get_document_stats(doc)
                    st.session_state.doc_loaded = False  # will rebuild index
                except Exception as e:
                    st.error(f"Parse error: {e}")

    # Show doc stats if loaded
    if st.session_state.parsed_doc:
        doc = st.session_state.parsed_doc
        fmt_icon = {"pdf": "📄", "docx": "📝", "txt": "📃"}.get(doc.format, "📋")
        st.markdown(f"""
        <div class="doc-badge">
            <span>{fmt_icon}</span>
            <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{doc.filename}</span>
        </div>
        """, unsafe_allow_html=True)

        stats = st.session_state.doc_stats
        cols = st.columns(3)
        raw_words = int(stats.get("words", "0").replace(",", ""))
        cols[0].metric("Words", format_number(raw_words))
        cols[1].metric("Sections", stats.get("sections", 0))
        cols[2].metric("Pages", stats.get("pages", "—"))

        if not st.session_state.doc_loaded:
            if not st.session_state.api_key_set and st.session_state.llm_backend != "ollama":
                st.warning("⚠️ Enter your API key first.")
            else:
                if st.button("⚡ Build Index & Start", use_container_width=True):
                    with st.spinner("Building vector index..."):
                        success = build_pipeline_from_state()
                    if success:
                        st.success(f"✓ Indexed {st.session_state.chunking_stats.get('total_chunks', '?')} chunks")
                    else:
                        err = st.session_state.get("_last_build_error", "Unknown error")
                        st.error(f"**Build failed:**\n\n{err}")

        if st.session_state.doc_loaded:
            cstats = st.session_state.chunking_stats
            st.markdown(f"""
            <div style="padding:0.75rem 0">
            <span class="stat-pill">⬡ {cstats.get('total_chunks','?')} chunks</span>
            <span class="stat-pill">≈ {cstats.get('avg_tokens_per_chunk','?')} avg tokens</span>
            </div>
            """, unsafe_allow_html=True)

    # Retrieval settings
    if st.session_state.doc_loaded:
        st.markdown('<span class="section-label">⚙️ Retrieval Settings</span>', unsafe_allow_html=True)
        with st.expander("Configure", expanded=False):
            st.session_state.top_k = st.slider("Top-K chunks", 1, 10, st.session_state.top_k)
            st.session_state.use_mmr = st.toggle("Maximal Marginal Relevance", st.session_state.use_mmr,
                                                  help="Balances relevance + diversity in retrieved chunks")
            st.session_state.similarity_threshold = st.slider(
                "Min similarity", 0.0, 0.8, st.session_state.similarity_threshold, 0.05
            )

        # Section filter
        if st.session_state.vector_store:
            sections = ["All sections"] + st.session_state.vector_store.get_stats().get("sections", [])
            selected = st.selectbox("Section filter", sections)
            st.session_state.selected_section = None if selected == "All sections" else selected

        # Reset chat
        if st.button("🗑 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.reset_memory()
            st.rerun()


# ════════════════════════════════════════════════
# MAIN CONTENT AREA
# ════════════════════════════════════════════════

# Top navigation bar
st.markdown("""
<div style="
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    height: 52px;
">
    <span style="font-family:var(--font-display);font-weight:700;font-size:1.1rem;color:var(--accent-light)">⬡ DocIQ</span>
    <span style="color:var(--border);font-size:1rem">|</span>
    <span style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.08em">Advanced Document Intelligence</span>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_summary, tab_extract, tab_search, tab_metrics = st.tabs([
    "💬 Chat",
    "📋 Summarize",
    "🔍 Extract",
    "🧭 Search",
    "📊 Metrics",
])


# ════════════════════════════════════════════════
# TAB 1: CHAT (RAG QA)
# ════════════════════════════════════════════════

with tab_chat:
    chat_wrapper = st.container()

    with chat_wrapper:
        # Messages display
        messages_container = st.container()
        with messages_container:
            if not st.session_state.doc_loaded:
                st.markdown("""
                <div class="welcome-screen">
                    <div class="welcome-icon">⬡</div>
                    <div class="welcome-title">DocIQ is ready</div>
                    <div class="welcome-sub">
                        Upload a document in the sidebar and build the index to start a conversation.
                        Ask anything — DocIQ grounds every answer in your document.
                    </div>
                    <div class="pill-grid">
                        <span class="example-pill">📄 PDF reports</span>
                        <span class="example-pill">📝 DOCX papers</span>
                        <span class="example-pill">📃 Text files</span>
                        <span class="example-pill">🔬 Research papers</span>
                        <span class="example-pill">📊 Technical docs</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show conversation messages
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="message-user">
                            <div class="message-user-bubble">{msg["content"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Render assistant message
                        content = msg["content"]
                        meta = msg.get("meta", {})

                        st.markdown(f"""
                        <div class="message-assistant">
                            <div class="assistant-icon">⬡</div>
                            <div>
                                <div class="message-assistant-bubble">
                        """, unsafe_allow_html=True)
                        st.markdown(content)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Metadata tags
                        if meta:
                            conf = meta.get("avg_confidence", 0)
                            conf_class = "good" if conf >= 0.75 else "med" if conf >= 0.5 else "low"
                            conf_label = f"{conf:.0%}"
                            sim = meta.get("top_similarity", 0)
                            total_ms = meta.get("total_time_ms", 0)

                            st.markdown(f"""
                            <div class="message-meta">
                                <span class="meta-tag {conf_class}">⬟ {conf_label} confidence</span>
                                <span class="meta-tag">≈ {sim:.2f} similarity</span>
                                <span class="meta-tag">⏱ {total_ms:.0f}ms</span>
                            </div>
                            """, unsafe_allow_html=True)

                            # Source chips
                            sources = meta.get("sources", [])
                            if sources:
                                chips = "".join([
                                    f'<span class="source-chip">§ {s["section"][:35]}</span>'
                                    for s in sources
                                ])
                                st.markdown(f'<div style="margin-top:6px">{chips}</div>',
                                            unsafe_allow_html=True)

                        st.markdown("</div></div>", unsafe_allow_html=True)

        # If welcome screen shown, add example question buttons
        if st.session_state.doc_loaded and not st.session_state.messages:
            st.markdown("**Try asking:**")
            example_qs = [
                "What is the main topic of this document?",
                "Summarize the key findings",
                "What methodology was used?",
                "What are the main conclusions?",
            ]
            cols = st.columns(2)
            for i, q in enumerate(example_qs):
                if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state["_auto_question"] = q
                    st.rerun()

        # Process auto question from example buttons
        if st.session_state.get("_auto_question") and st.session_state.doc_loaded:
            auto_q = st.session_state.pop("_auto_question")
            with st.spinner("Thinking..."):
                pipeline = st.session_state.rag_pipeline
                response = pipeline.answer(
                    question=auto_q,
                    top_k=st.session_state.top_k,
                    use_mmr=st.session_state.use_mmr,
                    similarity_threshold=st.session_state.similarity_threshold,
                    section_filter=st.session_state.selected_section,
                )
            st.session_state.messages.append({"role": "user", "content": auto_q})
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "meta": {
                    "avg_confidence": response.avg_confidence,
                    "top_similarity": response.top_similarity,
                    "total_time_ms": response.total_time_ms,
                    "sources": response.sources,
                },
            })
            st.session_state.last_response = response
            if st.session_state.evaluator:
                st.session_state.evaluator.record_query(response, auto_q)
            st.rerun()

        # Input bar
        st.markdown("---")
        if st.session_state.doc_loaded:
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                user_input = st.chat_input(
                    "Ask anything about your document...",
                    key="chat_input",
                )
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})

                with st.spinner("⬡ Retrieving and generating..."):
                    pipeline = st.session_state.rag_pipeline
                    response = pipeline.answer(
                        question=user_input,
                        top_k=st.session_state.top_k,
                        use_mmr=st.session_state.use_mmr,
                        similarity_threshold=st.session_state.similarity_threshold,
                        section_filter=st.session_state.selected_section,
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "meta": {
                        "avg_confidence": response.avg_confidence,
                        "top_similarity": response.top_similarity,
                        "total_time_ms": response.total_time_ms,
                        "sources": response.sources,
                    },
                })
                st.session_state.last_response = response

                if st.session_state.evaluator:
                    st.session_state.evaluator.record_query(response, user_input)

                st.rerun()

        else:
            st.info("⬡ Upload a document and build the index to start chatting.")

        # Retrieved chunks inspector (collapsible)
        if st.session_state.last_response:
            resp = st.session_state.last_response
            with st.expander(f"🔍 Retrieved Context ({len(resp.retrieved_chunks)} chunks)", expanded=False):
                for i, result in enumerate(resp.retrieved_chunks):
                    chunk = result.chunk
                    raw_text = chunk.metadata.get("raw_chunk_text", chunk.text)
                    conf_color = "#10b981" if result.confidence >= 0.75 else "#f59e0b" if result.confidence >= 0.5 else "#ef4444"

                    st.markdown(f"""
                    <div class="section-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                            <span style="font-family:var(--font-mono);font-size:0.75rem;color:var(--accent-light)">
                                § {chunk.section_title}
                            </span>
                            <span style="font-family:var(--font-mono);font-size:0.7rem;color:{conf_color}">
                                {result.confidence:.0%} confidence · {result.similarity_score:.3f} sim
                            </span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:{result.confidence*100:.0f}%;background:{conf_color}"></div>
                        </div>
                        <p style="font-size:0.82rem;color:var(--text-secondary);margin-top:8px;line-height:1.6">
                            {raw_text[:400]}{"..." if len(raw_text) > 400 else ""}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 2: SUMMARIZATION
# ════════════════════════════════════════════════

with tab_summary:
    st.markdown("""
    <div style="padding:1.5rem 2rem 0.5rem">
        <h3 style="font-family:var(--font-display);font-weight:700;font-size:1.3rem;color:var(--text-primary);margin:0">
            Multi-Mode Summarization
        </h3>
        <p style="color:var(--text-secondary);font-size:0.875rem;margin-top:4px">
            Generate different summary formats optimized for different audiences and use cases.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.doc_loaded:
        st.info("⬡ Load a document first to generate summaries.")
    else:
        col_mode, col_btn = st.columns([3, 1])

        with col_mode:
            summary_modes = {
                "tldr": "⚡ TL;DR — 2-3 sentence essence",
                "executive": "💼 Executive — Business-oriented with key findings",
                "technical": "🔬 Technical — Preserves methodology & metrics",
                "bullets": "• Bullet Points — Scannable key points",
                "hierarchical": "🏗 Hierarchical — Section-by-section + global synthesis",
            }
            mode_key = st.selectbox(
                "Summary Mode",
                list(summary_modes.keys()),
                format_func=lambda k: summary_modes[k],
                label_visibility="collapsed",
            )

        with col_btn:
            gen_btn = st.button("Generate Summary", use_container_width=True)

        if gen_btn:
            with st.spinner(f"Generating {mode_key} summary..."):
                pipeline = st.session_state.rag_pipeline
                doc = st.session_state.parsed_doc
                summary = pipeline.summarize(doc, mode=mode_key)
                st.session_state.current_summary = summary

        if st.session_state.current_summary:
            st.markdown("""
            <div style="
                background:var(--bg-card);
                border:1px solid var(--border);
                border-radius:var(--radius);
                padding:1.5rem;
                margin:1rem 0;
            ">
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.current_summary)
            st.markdown("</div>", unsafe_allow_html=True)

            col_copy, col_dl = st.columns([1, 1])
            with col_dl:
                st.download_button(
                    "⬇ Download Summary",
                    st.session_state.current_summary,
                    file_name=f"summary_{mode_key}.md",
                    mime="text/markdown",
                )


# ════════════════════════════════════════════════
# TAB 3: STRUCTURED EXTRACTION
# ════════════════════════════════════════════════

with tab_extract:
    st.markdown("""
    <div style="padding:1.5rem 2rem 0.5rem">
        <h3 style="font-family:var(--font-display);font-weight:700;font-size:1.3rem;color:var(--text-primary);margin:0">
            Structured Information Extraction
        </h3>
        <p style="color:var(--text-secondary);font-size:0.875rem;margin-top:4px">
            Extract named entities, key statistics, methodology, and conclusions as structured JSON.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.doc_loaded:
        st.info("⬡ Load a document first to extract structured data.")
    else:
        if st.button("⚗ Extract Structured Data", use_container_width=False):
            with st.spinner("Extracting entities and structured data..."):
                pipeline = st.session_state.rag_pipeline
                doc = st.session_state.parsed_doc
                extraction = pipeline.extract(doc)
                st.session_state.current_extraction = extraction

        if st.session_state.current_extraction:
            ext = st.session_state.current_extraction

            # Quick summary cards
            if isinstance(ext, dict) and "raw_extraction" not in ext:
                c1, c2, c3, c4 = st.columns(4)

                people = ext.get("named_entities", {}).get("people", [])
                orgs = ext.get("named_entities", {}).get("organizations", [])
                stats = ext.get("key_statistics", [])
                conclusions = ext.get("main_conclusions", [])

                c1.metric("People Found", len(people))
                c2.metric("Organizations", len(orgs))
                c3.metric("Key Statistics", len(stats))
                c4.metric("Conclusions", len(conclusions))

                col_left, col_right = st.columns(2)

                with col_left:
                    if people:
                        st.markdown("**👤 People**")
                        for p in people[:10]:
                            st.markdown(f"""<span class="stat-pill">👤 {p}</span>""",
                                        unsafe_allow_html=True)

                    if orgs:
                        st.markdown("**🏢 Organizations**")
                        for o in orgs[:10]:
                            st.markdown(f"""<span class="stat-pill">🏢 {o}</span>""",
                                        unsafe_allow_html=True)

                    locs = ext.get("named_entities", {}).get("locations", [])
                    if locs:
                        st.markdown("**📍 Locations**")
                        for l in locs[:8]:
                            st.markdown(f"""<span class="stat-pill">📍 {l}</span>""",
                                        unsafe_allow_html=True)

                with col_right:
                    if stats:
                        st.markdown("**📊 Key Statistics**")
                        for s in stats[:8]:
                            if isinstance(s, dict):
                                st.markdown(f"""
                                <div class="section-card" style="padding:0.6rem 0.9rem;margin-bottom:4px">
                                    <span style="color:var(--accent-light);font-family:var(--font-mono);font-size:0.8rem">
                                        {s.get('metric','')}: {s.get('value','')}
                                    </span>
                                    <p style="font-size:0.75rem;color:var(--text-dim);margin:2px 0 0">
                                        {s.get('context','')[:80]}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                    if conclusions:
                        st.markdown("**✅ Conclusions**")
                        for c in conclusions[:5]:
                            st.markdown(f"- {c}")

            # Full JSON view
            with st.expander("View raw JSON", expanded=False):
                st.code(json.dumps(ext, indent=2), language="json")

            st.download_button(
                "⬇ Download JSON",
                json.dumps(ext, indent=2),
                file_name="extraction.json",
                mime="application/json",
            )


# ════════════════════════════════════════════════
# TAB 4: SEMANTIC SEARCH
# ════════════════════════════════════════════════

with tab_search:
    st.markdown("""
    <div style="padding:1.5rem 2rem 0.5rem">
        <h3 style="font-family:var(--font-display);font-weight:700;font-size:1.3rem;color:var(--text-primary);margin:0">
            Section-Aware Semantic Search
        </h3>
        <p style="color:var(--text-secondary);font-size:0.875rem;margin-top:4px">
            Search document chunks by meaning — no keyword matching required.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.doc_loaded:
        st.info("⬡ Load a document to search.")
    else:
        col_search, col_top, col_btn2 = st.columns([4, 1, 1])
        with col_search:
            search_query = st.text_input(
                "Search",
                placeholder="Search document concepts, entities, or topics...",
                label_visibility="collapsed",
                key="search_input",
            )
        with col_top:
            search_k = st.selectbox("Top", [3, 5, 8, 10], index=1,
                                    label_visibility="collapsed")
        with col_btn2:
            search_btn = st.button("Search", use_container_width=True)

        if search_btn and search_query:
            with st.spinner("Searching..."):
                pipeline = st.session_state.rag_pipeline
                results = pipeline.semantic_search(
                    query=search_query,
                    top_k=search_k,
                )

            if not results:
                st.warning("No results found. Try a different query or lower the similarity threshold.")
            else:
                st.markdown(f"**{len(results)} results** for: *{search_query}*")
                st.markdown("---")

                for r in results:
                    chunk = r.chunk
                    raw_text = chunk.metadata.get("raw_chunk_text", chunk.text)
                    conf_pct = r.confidence * 100
                    conf_color = "#10b981" if r.confidence >= 0.75 else "#f59e0b" if r.confidence >= 0.5 else "#ef4444"

                    st.markdown(f"""
                    <div class="section-card">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                            <div>
                                <span style="font-family:var(--font-mono);font-size:0.75rem;color:var(--accent-light)">
                                    § {chunk.section_title}
                                </span>
                                <span style="margin-left:8px;font-family:var(--font-mono);font-size:0.65rem;color:var(--text-dim)">
                                    Rank #{r.rank + 1}
                                </span>
                            </div>
                            <span style="font-family:var(--font-mono);font-size:0.72rem;color:{conf_color};
                                         background:rgba(0,0,0,0.3);border-radius:4px;padding:2px 8px">
                                {r.similarity_score:.3f}
                            </span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:{conf_pct:.0f}%;background:{conf_color}"></div>
                        </div>
                        <p style="font-size:0.83rem;color:var(--text-secondary);margin-top:10px;line-height:1.65">
                            {raw_text[:500]}{"..." if len(raw_text) > 500 else ""}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 5: METRICS & EVALUATION
# ════════════════════════════════════════════════

with tab_metrics:
    st.markdown("""
    <div style="padding:1.5rem 2rem 0.5rem">
        <h3 style="font-family:var(--font-display);font-weight:700;font-size:1.3rem;color:var(--text-primary);margin:0">
            Performance Metrics & Evaluation
        </h3>
        <p style="color:var(--text-secondary);font-size:0.875rem;margin-top:4px">
            Real-time tracking of latency, retrieval quality, and system performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    evaluator = st.session_state.evaluator

    if evaluator is None or not evaluator.query_history:
        st.info("⬡ No queries recorded yet. Ask questions in the Chat tab to generate metrics.")
    else:
        stats = evaluator.get_aggregate_stats()

        # Latency metrics
        st.markdown("#### ⏱ Latency")
        c1, c2, c3, c4 = st.columns(4)
        lat = stats.get("latency", {})
        c1.metric("Avg Retrieval", f"{lat.get('avg_retrieval_ms', 0):.0f}ms")
        c2.metric("Avg Generation", f"{lat.get('avg_generation_ms', 0):.0f}ms")
        c3.metric("Avg Total", f"{lat.get('avg_total_ms', 0):.0f}ms")
        c4.metric("P95 Total", str(lat.get("p95_total_ms", "N/A")))

        # Retrieval quality
        st.markdown("#### 🎯 Retrieval Quality")
        c5, c6, c7, c8 = st.columns(4)
        rq = stats.get("retrieval_quality", {})
        c5.metric("Total Queries", stats.get("total_queries", 0))
        c6.metric("Avg Top Similarity", f"{rq.get('avg_top_similarity', 0):.3f}")
        c7.metric("Avg Confidence", f"{rq.get('avg_confidence', 0):.0%}")
        c8.metric("Avg Chunks/Query", f"{rq.get('avg_chunks_retrieved', 0):.1f}")

        # ROUGE scoring (vs reference)
        st.markdown("#### 📐 ROUGE Evaluation")
        with st.expander("Compute ROUGE Score (provide reference answer)", expanded=False):
            col_pred, col_ref = st.columns(2)
            with col_pred:
                if st.session_state.messages:
                    last_answer = next(
                        (m["content"] for m in reversed(st.session_state.messages)
                         if m["role"] == "assistant"), ""
                    )
                else:
                    last_answer = ""
                pred_text = st.text_area("Prediction (last answer)", value=last_answer[:500], height=150)
            with col_ref:
                ref_text = st.text_area("Reference (ground truth)", height=150,
                                        placeholder="Enter the expected/correct answer...")

            if st.button("Compute ROUGE") and pred_text and ref_text:
                rouge_scores = evaluator.compute_rouge(pred_text, ref_text)
                st.json(rouge_scores)

        # Per-query table
        st.markdown("#### 📋 Query History")
        table_data = evaluator.get_per_query_table()
        if table_data:
            st.dataframe(table_data, use_container_width=True)

    # System info
    st.markdown("#### 🖥 System Configuration")
    if st.session_state.embedding_engine:
        info = st.session_state.embedding_engine.get_model_info()
        col_a, col_b = st.columns(2)
        with col_a:
            st.code(f"""
Embedding Model : {info['model_name']}
Embedding Dim   : {info['embedding_dim']}
LLM Backend     : Anthropic Claude
LLM Model       : claude-sonnet-4-20250514
            """, language="text")
        with col_b:
            vs = st.session_state.vector_store
            if vs:
                vs_stats = vs.get_stats()
                st.code(f"""
Vector DB       : FAISS IndexFlatIP
Chunks Indexed  : {vs_stats['total_chunks']}
Docs Indexed    : {vs_stats['documents_indexed']}
Index Type      : {vs_stats['index_type']}
            """, language="text")
    else:
        st.info("Load a document to see system configuration.")
