# ⬡ DocIQ — Advanced Document Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Upload any document. Ask anything. Get answers grounded in your content.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Deployment](#-deployment) • [Models](#-models-used)

</div>

---

## 📌 What is DocIQ?

DocIQ is a **production-grade Retrieval-Augmented Generation (RAG) system** that transforms static documents into an interactive conversational intelligence layer. Upload a PDF, DOCX, or TXT file and instantly chat with it, summarize it, extract structured data, or search it semantically — all with answers strictly grounded in your document content.

> **No hallucination. No guessing. Every answer comes from your document.**

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Conversational QA** | Chat with your document like a conversation with full memory |
| 📋 **Multi-Mode Summarization** | TL;DR, Executive, Technical, Bullet Points, Hierarchical |
| 🔍 **Structured Extraction** | Auto-extract entities, statistics, conclusions as JSON |
| 🧭 **Semantic Search** | Find relevant sections by meaning, not keywords |
| 📊 **Evaluation Metrics** | ROUGE scores, latency tracking, retrieval confidence |
| 🔒 **Anti-Hallucination** | LLM strictly restricted to retrieved document context |
| 🧠 **Conversation Memory** | Sliding window + LLM-compressed memory across turns |
| 📎 **Multi-Format Support** | PDF, DOCX, TXT, Markdown |
| 🦙 **100% Local Option** | Run with Ollama — no API key, no internet, no cost |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DocIQ Pipeline                        │
│                                                          │
│  DOCUMENT                                                │
│  UPLOAD    →  Parser  →  Chunker  →  Embedder  →  FAISS │
│                                           ↑              │
│                                    MiniLM-L6-v2          │
│                                                          │
│  USER                                                    │
│  QUESTION  →  Embed Query  →  FAISS Search  →  Top-K    │
│                                                   ↓      │
│                              Memory  →  Prompt Builder   │
│                                                   ↓      │
│                                         LLM (Mistral /   │
│                                         Claude / GPT)    │
│                                                   ↓      │
│                                            ANSWER ✅     │
└─────────────────────────────────────────────────────────┘
```

### Core Components

```
dociq/
├── app.py                      # Streamlit UI — chat, summarize, extract, search
├── config.py                   # Typed configuration for all system parameters
├── requirements.txt
├── core/
│   ├── document_processor.py   # PDF / DOCX / TXT parsing with section detection
│   ├── chunker.py              # Token-aware + section-aware chunking (512 tok, 64 overlap)
│   ├── embeddings.py           # Sentence-transformers embedding engine
│   ├── vector_store.py         # FAISS index + MMR retrieval + confidence scoring
│   ├── llm_engine.py           # Multi-provider LLM (Anthropic / OpenAI / Ollama / Groq)
│   ├── rag_pipeline.py         # End-to-end RAG orchestrator + conversation memory
│   └── evaluator.py            # ROUGE, latency, retrieval quality metrics
├── prompts/
│   └── templates.py            # Prompt library for QA, summarization, extraction
└── utils/
    └── session.py              # Streamlit session state management
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- [Ollama](https://ollama.com) (for local free usage)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOURNAME/dociq.git
cd dociq
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set up LLM backend

**Option A — Ollama (Free, Local, Recommended)**
```bash
# Install Ollama from https://ollama.com
ollama pull mistral:7b        # or llama3.2:3b for faster/lighter

# Start Ollama server
ollama serve
```

**Option B — Anthropic Claude**
```bash
# Create .env file
echo ANTHROPIC_API_KEY=sk-ant-your-key-here > .env
```

**Option C — Groq (Free API)**
```bash
# Sign up free at console.groq.com
echo GROQ_API_KEY=your-key-here > .env
```

### Step 5 — Launch
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥 Usage

### Basic Workflow
1. **Select LLM backend** in the sidebar (Ollama / Anthropic / OpenAI)
2. **Upload a document** (PDF, DOCX, or TXT)
3. Click **"⚡ Build Index & Start"** — embeds and indexes your document
4. **Start chatting** in the Chat tab

### Chat Tab
Ask anything about your document:
```
"What is the main conclusion of this paper?"
"What methodology was used?"
"Summarize the results section"
"Who are the authors?"
```

### Summarize Tab
Choose a mode:
- **TL;DR** — 2-3 sentence essence
- **Executive** — Business-oriented with key findings
- **Technical** — Preserves methodology and metrics
- **Bullet Points** — Scannable key points
- **Hierarchical** — Section-by-section then global synthesis

### Extract Tab
Auto-extracts structured JSON:
```json
{
  "title": "...",
  "authors": ["..."],
  "named_entities": {
    "people": ["..."],
    "organizations": ["..."],
    "locations": ["..."]
  },
  "key_statistics": [...],
  "main_conclusions": [...]
}
```

### Search Tab
Semantic search across all document chunks — finds by meaning not keywords.

### Metrics Tab
Real-time performance dashboard:
- Retrieval latency, generation latency, P95 latency
- Top cosine similarity, average confidence
- ROUGE-1, ROUGE-2, ROUGE-L scores

---

## 🤖 Models Used

| Model | Role | Size | Provider |
|---|---|---|---|
| `all-MiniLM-L6-v2` | Text → Embeddings (384-dim) | 80MB | Microsoft / HuggingFace |
| `mistral:7b` | LLM — Answer generation | 4.1GB | Mistral AI via Ollama |
| `llama3.1:8b` | LLM — Alternative | 4.7GB | Meta via Ollama |
| `claude-sonnet-4` | LLM — Cloud option | API | Anthropic |
| `FAISS IndexFlatIP` | Vector similarity search | — | Meta |

---

## ⚙️ Configuration

All settings in `config.py`:

```python
# Chunking
chunk_size = 512          # tokens per chunk
chunk_overlap = 64        # overlap between chunks

# Retrieval
top_k = 5                 # chunks to retrieve
similarity_threshold = 0.20
use_mmr = True            # Maximal Marginal Relevance
mmr_lambda = 0.7          # relevance vs diversity balance

# Memory
memory_window = 10        # last N conversation turns to keep
```

---

## 🌐 Deployment

### Streamlit Cloud (Free, Recommended)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as main file
4. Add secret: `ANTHROPIC_API_KEY = "sk-ant-..."`
5. Deploy → get public URL

### Groq + Streamlit Cloud (Completely Free)
1. Get free API key at [console.groq.com](https://console.groq.com)
2. Deploy on Streamlit Cloud with `GROQ_API_KEY` secret
3. Zero cost, public URL, production ready

### Oracle Cloud Free VM (Always On, No API Key)
```bash
# On Oracle free VM (4 CPU, 24GB RAM)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
git clone https://github.com/YOURNAME/dociq.git
cd dociq && pip install -r requirements.txt
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Quick Demo (ngrok)
```bash
# Run app locally then expose publicly
streamlit run app.py
ngrok http 8501
# Share the ngrok URL
```

---

## 🔬 Technical Details

### Chunking Strategy
- **Phase 1:** Split by document sections (headings, numbered sections) — never breaks logical units
- **Phase 2:** Token-aware sliding window within each section (512 tokens, 64 overlap)
- **Phase 3:** Metadata enrichment — section title prepended to chunk for better embedding quality

### Retrieval Strategy
```
FAISS k-NN (k = top_k × 3)
    → Similarity threshold filter
    → MMR reranking for diversity
    → Token budget management (5000 token max context)
    → Confidence score normalization
```

### MMR Formula
```
MMR(d) = λ × sim(query, d) − (1−λ) × max{sim(d, s) : s ∈ Selected}
```

### Anti-Hallucination
Every prompt contains:
```
Answer EXCLUSIVELY from the provided document context.
Do NOT use external knowledge.
If the answer is not found, say: "I cannot find this in the document."
```

### Conversation Memory
```
Recent turns (last 10) → verbatim in prompt
Older turns (>20)      → LLM-compressed summary
```

---

## 📊 Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| ROUGE-1 F1 | Unigram overlap with reference | > 0.40 |
| ROUGE-2 F1 | Bigram overlap | > 0.20 |
| ROUGE-L F1 | Longest common subsequence | > 0.35 |
| Retrieval Latency | FAISS search time | < 50ms |
| Generation Latency | LLM response time | < 3000ms |
| Top Similarity | Best chunk cosine score | > 0.55 |

---

## 🛡 Security Notes

- Never commit `.env` files or API keys to GitHub
- Add `.env` and `*.log` to `.gitignore`
- For sensitive documents use local Ollama — data never leaves your machine
- API keys shown in sidebar are stored in session state only — not persisted to disk

---

## 🔮 Roadmap

- [ ] Cross-encoder reranker (BGE-reranker-large)
- [ ] OCR support for scanned PDFs (Tesseract)
- [ ] Multi-document indexing with source filtering
- [ ] BERTScore evaluation
- [ ] FastAPI backend for production
- [ ] Pinecone / Qdrant cloud vector DB support
- [ ] Export chat history as PDF

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Built With

- [Streamlit](https://streamlit.io) — Web UI framework
- [Sentence Transformers](https://sbert.net) — Embedding models
- [FAISS](https://faiss.ai) — Vector similarity search
- [Ollama](https://ollama.com) — Local LLM runner
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF extraction
- [Anthropic](https://anthropic.com) — Claude API

---

## Author
- *Aadil Hussain*

<div align="center">
Built with ❤️ as a portfolio project demonstrating production RAG architecture
</div>
