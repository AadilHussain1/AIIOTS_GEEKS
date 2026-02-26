# 📁 Project Structure

```
resume_screener/
│
├── 📄 app.py                          # Main Streamlit application (500+ lines)
│   ├── UI Components
│   │   ├── Gradient background design
│   │   ├── Custom CSS styling
│   │   ├── File upload interface
│   │   └── Metric cards & visualizations
│   ├── Core Functions
│   │   ├── BERT model loading
│   │   ├── Text extraction (PDF/DOCX/TXT)
│   │   ├── Semantic similarity analysis
│   │   ├── Skill extraction & matching
│   │   └── TF-IDF keyword analysis
│   ├── Visualizations
│   │   ├── Gauge chart (match score)
│   │   ├── Pie chart (skill distribution)
│   │   └── Bar chart (skill comparison)
│   └── Export Functions
│       ├── PDF export
│       ├── TXT export
│       └── CSV export
│
├── 📋 requirements.txt                # Python dependencies
│   ├── streamlit (UI framework)
│   ├── sentence-transformers (BERT)
│   ├── PyPDF2 (PDF processing)
│   ├── python-docx (Word processing)
│   ├── plotly (visualizations)
│   ├── pandas & numpy (data handling)
│   └── fpdf (PDF generation)
│
├── 📖 README.md                       # Comprehensive documentation
│   ├── Features overview
│   ├── Installation guide
│   ├── Usage instructions
│   ├── Technology stack
│   ├── Troubleshooting
│   └── Future enhancements
│
├── 🚀 QUICKSTART.md                   # Quick start guide
│   ├── 5-minute setup
│   ├── First use tutorial
│   ├── Testing with samples
│   └── Common issues
│
├── 🔧 setup.sh                        # Linux/Mac setup script
│   ├── Python version check
│   ├── Virtual environment creation
│   ├── Dependency installation
│   └── BERT model download
│
├── 🔧 setup.bat                       # Windows setup script
│   └── Same functionality as setup.sh
│
├── 📝 sample_resume.txt              # Sample resume for testing
│   └── Complete candidate profile
│
├── 📝 sample_job_description.txt     # Sample JD for testing
│   └── Complete job posting
│
└── 🚫 .gitignore                      # Git ignore file
    ├── Python cache files
    ├── Virtual environments
    ├── Models (auto-downloaded)
    └── User uploads/exports
```

## 🎨 UI Components Breakdown

### Main Page
```
┌─────────────────────────────────────────┐
│  🎯 AI Resume Screener                  │
│  Powered by Advanced BERT Technology    │
├─────────────────────────────────────────┤
│                                          │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ 📄 Upload    │  │ 💼 Job       │    │
│  │   Resume     │  │  Description │    │
│  └──────────────┘  └──────────────┘    │
│                                          │
│      [🚀 Analyze Resume]                │
│                                          │
└─────────────────────────────────────────┘
```

### Results Page
```
┌─────────────────────────────────────────┐
│  📊 Analysis Results                    │
├─────────────────────────────────────────┤
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│  │Match │ │Matched│ │Missing│ │Status│  │
│  │Score │ │Skills │ │Skills │ │      │  │
│  └──────┘ └──────┘ └──────┘ └──────┘  │
│                                          │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ Gauge Chart  │  │  Pie Chart   │    │
│  └──────────────┘  └──────────────┘    │
│                                          │
│  ┌─────────────────────────────────┐   │
│  │     Bar Chart (Skills)          │   │
│  └─────────────────────────────────┘   │
│                                          │
│  ┌──────────────┐  ┌──────────────┐    │
│  │  ✅ Matched  │  │  ❌ Missing  │    │
│  │    Skills    │  │    Skills    │    │
│  └──────────────┘  └──────────────┘    │
│                                          │
│  [📄 PDF] [📝 TXT] [📊 CSV]            │
└─────────────────────────────────────────┘
```

## 🔄 Data Flow

```
User Input
    │
    ├─> Resume File (PDF/DOCX/TXT)
    │       │
    │       └─> Text Extraction
    │               │
    └─> Job Description (File/Text)
            │
            └─> Text Processing
                    │
                    ▼
            BERT Model (Sentence Transformers)
                    │
                    ├─> Semantic Similarity (Cosine)
                    ├─> Skill Extraction
                    ├─> TF-IDF Keywords
                    │
                    ▼
            Analysis Results
                    │
                    ├─> Match Score (0-100%)
                    ├─> Recommendation
                    ├─> Matched Skills
                    ├─> Missing Skills
                    │
                    ▼
            Visualizations
                    │
                    ├─> Gauge Chart
                    ├─> Pie Chart
                    ├─> Bar Chart
                    │
                    ▼
            Export Options
                    │
                    ├─> PDF Report
                    ├─> TXT Summary
                    └─> CSV Data
```

## 🎯 Key Features by File

| File | Primary Features |
|------|-----------------|
| **app.py** | Main application, UI, analysis logic, visualizations, exports |
| **requirements.txt** | All Python dependencies with versions |
| **README.md** | Complete documentation and guide |
| **QUICKSTART.md** | Fast setup and first-use tutorial |
| **setup.sh/.bat** | Automated installation scripts |
| **sample files** | Test data for immediate use |

## 💾 File Sizes (Approximate)

- **app.py**: ~20 KB (500+ lines of code)
- **requirements.txt**: ~0.3 KB
- **README.md**: ~7 KB
- **Downloaded BERT model**: ~80 MB (auto-downloaded)
- **Total project**: ~100 MB (with model)

## 🔧 Technology Stack Details

### Frontend (UI)
- **Streamlit**: Web interface
- **Custom CSS**: Gradient design, glassmorphism
- **Plotly**: Interactive charts

### Backend (Processing)
- **Sentence-BERT**: Semantic similarity
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **Scikit-learn**: TF-IDF analysis

### Data & Exports
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **FPDF**: PDF report generation

## 🚀 Deployment Options

1. **Local Development**: `streamlit run app.py`
2. **Streamlit Cloud**: Deploy via GitHub
3. **Docker**: Containerized deployment
4. **Heroku**: Cloud hosting
5. **AWS/Azure**: Enterprise deployment

---

**Total Lines of Code**: ~500+ (app.py)
**Setup Time**: 5 minutes
**First Analysis**: <10 seconds
