# DocIQ — Advanced Document Intelligence System
## Complete Technical Architecture Blueprint

> **Version:** 1.0.0 | **Stack:** Python 3.11 · Streamlit · FAISS · Sentence-Transformers · Anthropic Claude  
> **Architecture Pattern:** RAG (Retrieval-Augmented Generation) + Hierarchical Pipeline

---

## 1. System Architecture Overview

DocIQ is a **modular, production-ready RAG system** that transforms static documents into an interactive intelligence layer. Every component is independently testable and replaceable.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DocIQ System Architecture                        │
│                                                                          │
│   ┌──────────┐    ┌────────────────────────────────────────────────┐    │
│   │  Streamlit│   │              Ingestion Pipeline                │    │
│   │   Web UI  │───▶│  DocumentProcessor → Chunker → EmbeddingEngine │    │
│   │(app.py)  │    │         ↓                                       │    │
│   └──────────┘    │    VectorStore (FAISS)                          │    │
│         │          └────────────────────────────────────────────────┘    │
│         │                           │                                     │
│         │          ┌────────────────▼───────────────────────────────┐    │
│         │          │              Query Pipeline                      │    │
│         │          │                                                  │    │
│         │    Query │  ┌───────────┐   ┌──────────┐   ┌───────────┐  │    │
│         └─────────▶│  │  Memory   │──▶│Retrieval │──▶│  LLM Gen  │  │    │
│                    │  │ (sliding  │   │(MMR+FAISS│   │(Anthropic/│  │    │
│                    │  │  window)  │   │cosine)   │   │OpenAI/    │  │    │
│                    │  └───────────┘   └──────────┘   │Ollama)    │  │    │
│                    │                                  └───────────┘  │    │
│                    │         ┌──────────────────────────────────┐    │    │
│                    │         │           Task Router             │    │    │
│                    │         │  QA │ Summarize │ Extract │ Search│    │    │
│                    │         └──────────────────────────────────┘    │    │
│                    └────────────────────────────────────────────────┘    │
│                                        │                                  │
│                    ┌───────────────────▼────────────────────────────┐    │
│                    │           Evaluation Engine                      │    │
│                    │    ROUGE · Latency · Retrieval Score · P95       │    │
│                    └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component-Level Breakdown

### 2.1 DocumentProcessor (`core/document_processor.py`)

**Responsibility:** Multi-format text extraction with structure preservation.

| Component | Role |
|-----------|------|
| `PDFParser` | pdfplumber (primary) → PyPDF2 (fallback). Layout-aware extraction. |
| `DOCXParser` | python-docx with heading style detection and metadata extraction. |
| `TXTParser` | Multi-encoding detection (UTF-8 → UTF-8-sig → latin-1 → cp1252). |
| `BaseParser._detect_sections()` | Regex-based section detection across markdown, numbered, ALL-CAPS, and academic header patterns. |
| `BaseParser._clean_text()` | Removes control characters, normalizes unicode, collapses excessive whitespace. |

**Output:** `ParsedDocument` dataclass with `raw_text`, `sections[]`, `metadata`, `page_count`.

---

### 2.2 DocumentChunker (`core/chunker.py`)

**Responsibility:** Token-aware, section-respecting text segmentation.

**Three-Phase Strategy:**
```
Phase 1: SECTION SPLIT
  └─ Use DocumentSection boundaries (never split across sections)

Phase 2: TOKEN-AWARE SPLIT within each section
  └─ Target: 512 tokens/chunk, 64-token overlap
  └─ Sentence boundary awareness (no mid-sentence cuts)
  └─ Fallback word-level split for sentences > max_tokens

Phase 3: METADATA ENRICHMENT
  └─ Assign section_title, section_level, chunk_index
  └─ Generate deterministic SHA-256 chunk_id
  └─ Inject section header prefix for embedding quality
```

**Token Counting:** tiktoken (cl100k_base, accurate) → heuristic fallback (chars × 0.75).

---

### 2.3 EmbeddingEngine (`core/embeddings.py`)

**Responsibility:** Dense vector generation for semantic retrieval.

**Model Selection Rationale:**

| Model | Dim | Size | Quality | Speed | Use Case |
|-------|-----|------|---------|-------|----------|
| `all-MiniLM-L6-v2` | 384 | 80MB | ★★★☆ | ★★★★★ | **Default** — fast, efficient |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | ★★★★★ | ★★★☆ | Best quality, needs RAM |
| `all-mpnet-base-v2` | 768 | 420MB | ★★★★ | ★★★★ | Balanced option |

**Key Design Decisions:**
- L2-normalized embeddings → cosine similarity = dot product (faster FAISS search)
- Batch encoding (64 chunks/batch) for throughput
- BGE query prefix injection for asymmetric retrieval
- Streamlit `@st.cache_resource` → model loads **once per session**

---

### 2.4 VectorStore (`core/vector_store.py`)

**Responsibility:** FAISS-backed semantic retrieval with MMR diversity.

**Index Selection:**
```
chunks ≤ 50,000  →  IndexFlatIP  (exact search, O(n), perfect for documents)
chunks > 50,000  →  IndexIVFFlat (approximate, O(√n), nlist=256, nprobe=32)
```

**Retrieval Pipeline:**
```
Query Embedding
      ↓
FAISS k-NN (k = top_k × 3 for pre-filter)
      ↓
Similarity Threshold Filter (default: 0.20 cosine)
      ↓
Section Filter (optional: restrict to specific section)
      ↓
MMR Reranking (λ=0.7, balances relevance vs diversity)
      ↓
Confidence Score Normalization (relative to top result)
      ↓
RetrievalResult[] with metadata
```

**Maximal Marginal Relevance (MMR):**
```
MMR(d) = λ × sim(query, d) − (1−λ) × max{sim(d, s) : s ∈ Selected}
```
Prevents returning 5 near-identical chunks about the same paragraph.

---

### 2.5 LLMEngine (`core/llm_engine.py`)

**Responsibility:** Provider-agnostic LLM abstraction with streaming support.

**Provider Hierarchy:**
```python
BaseLLMProvider (ABC)
    ├── AnthropicProvider   → claude-sonnet-4-20250514 (primary, streaming)
    ├── OpenAIProvider      → gpt-4o (alternative)
    └── OllamaProvider      → llama3.2:3b / mistral:7b (local/offline)
```

**Open-Source LLM Recommendations (Ollama):**

| Model | Params | VRAM | Quality | Best For |
|-------|--------|------|---------|----------|
| `llama3.2:3b` | 3B | 3GB | ★★★☆ | Fast, resource-constrained |
| `mistral:7b` | 7B | 6GB | ★★★★ | Balanced default |
| `phi3:mini` | 3.8B | 4GB | ★★★☆ | Microsoft, efficient |
| `gemma2:9b` | 9B | 9GB | ★★★★★ | Best open-source quality |
| `deepseek-r1:7b` | 7B | 7GB | ★★★★ | Reasoning-heavy tasks |

---

### 2.6 Prompt Engineering (`prompts/templates.py`)

**Anti-Hallucination Architecture:**

Every prompt contains this critical grounding instruction:
```
CRITICAL: Answer EXCLUSIVELY from the provided document context.
Do NOT use external knowledge or training data.
If the answer is not found, respond: "I cannot find this information."
```

**Task-Specific Prompts:**

| Task | Temperature | Max Tokens | Strategy |
|------|-------------|------------|----------|
| QA | 0.1 | 2000 | Low temp for factual grounding |
| Summarization | 0.2 | 1500 | Slightly higher for fluency |
| Extraction | 0.0 | 2000 | Zero temp for deterministic JSON |
| Hierarchical | 0.2 | 2000 | Section-by-section + synthesis |

**Context Assembly Pattern:**
```
[Context 1 | Section: "Methodology" | Confidence: 87%]
<chunk text>

[Context 2 | Section: "Results" | Confidence: 71%]
<chunk text>

PREVIOUS CONVERSATION:
User: What methodology was used?
Assistant: The paper uses...

USER QUESTION: <current query>
```

---

### 2.7 RAGPipeline (`core/rag_pipeline.py`)

**Responsibility:** Orchestrates the complete retrieval-generation flow.

**Token Budget Management:**
```python
def _apply_token_budget(results, max_tokens=5000):
    selected = []
    used_tokens = 0
    for result in results:  # sorted by relevance
        if used_tokens + result.chunk.token_count <= max_tokens:
            selected.append(result)
            used_tokens += result.chunk.token_count
        else:
            break  # stop before exceeding budget
    return selected or [results[0]]  # always include at least top-1
```

**Conversation Memory:**
```
Sliding Window (last 10 turns verbatim)
    +
LLM Compression (when turns > 20)
    → Compressed summary of earlier conversation
    → Injected as context prefix: "[Earlier summary: ...]"
```

---

### 2.8 EvaluationEngine (`core/evaluator.py`)

**Metrics Tracked:**

| Metric | Description | Target |
|--------|-------------|--------|
| Retrieval Latency | FAISS search time (ms) | < 50ms |
| Generation Latency | LLM response time (ms) | < 3000ms |
| Top Similarity | Cosine score of best chunk | > 0.55 |
| Avg Confidence | Mean normalized confidence | > 0.65 |
| ROUGE-1 F1 | Unigram overlap with reference | > 0.40 |
| ROUGE-2 F1 | Bigram overlap with reference | > 0.20 |
| ROUGE-L F1 | Longest common subsequence | > 0.35 |
| P95 Latency | 95th percentile total time | < 5000ms |

---

## 3. Data Flow Diagram

```
USER UPLOADS DOCUMENT
         │
         ▼
┌─────────────────────────────────────┐
│     INGESTION PIPELINE              │
│                                     │
│  file_bytes + filename              │
│         ↓                           │
│  DocumentProcessor.process()        │
│  ├─ Format detection (.pdf/.docx/.txt)
│  ├─ Text extraction                 │
│  ├─ Section detection               │
│  └─ ParsedDocument ──────────────┐  │
│                                  │  │
│  DocumentChunker.chunk_document()│  │
│  ├─ Section-aware splitting      │  │
│  ├─ Token-aware sliding window   │  │
│  └─ DocumentChunk[] ─────────┐  │  │
│                               │  │  │
│  EmbeddingEngine.embed_chunks()  │  │
│  └─ np.ndarray (N, 384) ─────┘  │  │
│                                  │  │
│  VectorStore.build_index()       │  │
│  └─ FAISS IndexFlatIP ◀──────── ─┘  │
└─────────────────────────────────────┘
                    │
                    ▼ (index ready)
USER ASKS QUESTION
         │
         ▼
┌─────────────────────────────────────────────────────┐
│     QUERY PIPELINE                                   │
│                                                      │
│  ConversationMemory.get_history_string()             │
│         ↓                                            │
│  EmbeddingEngine.embed_query(question)               │
│  └─ query_vec: np.ndarray (384,)                     │
│         ↓                                            │
│  VectorStore.retrieve()                              │
│  ├─ FAISS.search(query_vec, k=15)                    │
│  ├─ Similarity threshold filter                      │
│  ├─ MMR reranking (λ=0.7)                            │
│  └─ RetrievalResult[] (top-5)                        │
│         ↓                                            │
│  RAGPipeline._apply_token_budget()                   │
│  └─ Filtered results (≤ 5000 tokens total)           │
│         ↓                                            │
│  build_rag_qa_prompt(question, chunks, history)      │
│  └─ (system_prompt, user_message)                    │
│         ↓                                            │
│  LLMEngine.generate_stream()                         │
│  └─ token stream ──▶ Streamlit UI                    │
│         ↓                                            │
│  ConversationMemory.add_turn()                       │
│  EvaluationEngine.record_query()                     │
│         ↓                                            │
│  RAGResponse(answer, chunks, metrics)                │
└─────────────────────────────────────────────────────┘
```

---

## 4. Chunking Strategy Design

### Why Section-First Chunking?

Standard recursive character splitting ignores document structure. DocIQ uses a **two-phase approach**:

1. **Section boundaries are sacred.** A chunk never crosses a section boundary. This preserves the semantic coherence of headings like "Results" or "Conclusion".

2. **Token-aware within sections.** After section splitting, sliding window token chunking ensures LLM context windows aren't exceeded.

3. **Overlap for context continuity.** 64-token overlap ensures that information split across chunk boundaries isn't lost during retrieval.

4. **Metadata enrichment.** Section title is prepended to chunk text during embedding: `"Methodology: We used a 5-fold cross-validation..."` This dramatically improves retrieval precision for section-specific queries.

---

## 5. Retrieval Strategy Design

### Multi-Stage Retrieval
```
Stage 1: Broad Recall
  FAISS retrieves k×3 candidates (over-retrieve for filtering)

Stage 2: Quality Filter
  Remove results below similarity_threshold (default 0.20)

Stage 3: MMR Diversity
  Rerank with Maximal Marginal Relevance to avoid redundant chunks

Stage 4: Token Budget
  Trim to fit within context window (5000 tokens max)
```

### Confidence Scoring Formula
```python
confidence = similarity_score / max_similarity_in_results
# Normalized relative score: top result always = 1.0
# Downstream results rated relative to top
```

---

## 6. Hardware Requirements

### Minimum (CPU-only)
- **RAM:** 8GB (4GB for MiniLM + 2GB app + 2GB OS)
- **Disk:** 2GB (models + FAISS index)
- **CPU:** 4 cores recommended
- **Expected latency:** ~3-8 seconds per query

### Recommended (Production)
- **RAM:** 16GB+
- **GPU:** NVIDIA with 8GB+ VRAM (for GPU-accelerated FAISS + torch)
- **Disk:** 10GB SSD
- **Expected latency:** < 1 second retrieval, 1-3 seconds generation

### Optimization Strategies
| Strategy | Impact | Implementation |
|----------|--------|----------------|
| Model quantization | 2-4× memory reduction | `model.half()` or bitsandbytes |
| Batch embedding | 3-5× embedding throughput | `batch_size=64` |
| FAISS GPU | 10-100× retrieval speed | `faiss-gpu` package |
| IndexIVFFlat | Faster approximate search | For >50k chunks |
| Streamlit cache | Eliminates model reload | `@st.cache_resource` |

---

## 7. Project Structure

```
dociq/
├── app.py                    # Main Streamlit application
├── config.py                 # Typed configuration dataclasses
├── requirements.txt          # Pinned dependencies
│
├── core/
│   ├── document_processor.py # Multi-format text extraction
│   ├── chunker.py            # Token-aware section chunker
│   ├── embeddings.py         # Sentence-transformer wrapper
│   ├── vector_store.py       # FAISS + MMR retrieval
│   ├── llm_engine.py         # Multi-provider LLM abstraction
│   ├── rag_pipeline.py       # End-to-end RAG orchestrator
│   └── evaluator.py          # ROUGE + latency metrics
│
├── prompts/
│   └── templates.py          # Task-specific prompt library
│
└── utils/
    └── session.py            # Streamlit session management
```

---

## 8. Scalability Roadmap

| Scale | Solution |
|-------|----------|
| **Multiple docs** | `VectorStore.add_chunks()` incremental indexing |
| **Large corpora (>100k chunks)** | Switch to `IndexIVFFlat` with GPU FAISS |
| **Multi-user** | Redis session store + shared FAISS index |
| **Production deployment** | FastAPI backend + separate embedding service |
| **Enterprise scale** | Pinecone/Weaviate/Qdrant cloud vector DB |
| **Reranking** | Add cross-encoder reranker (BAAI/bge-reranker-large) |
| **Async processing** | Celery task queue for document ingestion |
| **Streaming at scale** | Server-Sent Events via FastAPI |

---

## 9. Getting Started

```bash
# 1. Clone and install
git clone <repo>
cd dociq
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
# Or create .env file: ANTHROPIC_API_KEY=sk-ant-...

# 3. Launch
streamlit run app.py

# 4. Optional: Local LLM via Ollama
ollama serve
ollama pull mistral:7b
# Then select "ollama" backend in config.py
```

---

## 10. Advanced Features Implemented

| Feature | Implementation | Location |
|---------|---------------|----------|
| Hierarchical Summarization | Section-by-section → global synthesis | `rag_pipeline.py:_hierarchical_summarize()` |
| Section-Aware Retrieval | Section title in embedding + section filter | `chunker.py`, `vector_store.py` |
| Named Entity Extraction | Structured JSON via zero-temp LLM | `rag_pipeline.py:extract()` |
| Confidence Scoring | Normalized cosine relative to top result | `vector_store.py:_compute_confidence()` |
| Conversational Memory | Sliding window + LLM compression | `rag_pipeline.py:ConversationMemory` |
| Multi-mode Summarization | 5 modes: TLDR/Executive/Technical/Bullets/Hierarchical | `prompts/templates.py` |
| MMR Diversity | Carbonell & Goldstein 1998 algorithm | `vector_store.py:_mmr()` |
| Token Budget Management | Per-query context window enforcement | `rag_pipeline.py:_apply_token_budget()` |
| ROUGE Evaluation | rouge-score package, user-provided reference | `evaluator.py:compute_rouge()` |
| Latency Tracking | Per-query retrieval + generation timing | `evaluator.py:record_query()` |
| Multi-provider LLM | Anthropic / OpenAI / Ollama switchable | `llm_engine.py` |
| Multi-format Parsing | PDF (pdfplumber+PyPDF2) / DOCX / TXT | `document_processor.py` |
| Streaming Generation | Token-by-token via Anthropic stream API | `llm_engine.py:generate_stream()` |

---

*Built as a portfolio-grade AI system demonstrating production RAG architecture.*  
*DocIQ*

## 11. Author
*Aadil Hussain* 
