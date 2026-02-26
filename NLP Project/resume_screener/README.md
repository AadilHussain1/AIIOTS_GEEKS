# 🎯 AI-Powered Resume Screener

An intelligent resume screening application powered by BERT (Sentence Transformers) that analyzes resume-job description compatibility with beautiful visualizations and comprehensive insights.

## ✨ Features

- **Multi-format Support**: Upload resumes in PDF, DOCX, or TXT formats
- **AI-Powered Analysis**: Uses BERT-based semantic similarity for accurate matching
- **Smart Skill Extraction**: Automatically identifies and matches technical and soft skills
- **Interactive Visualizations**: 
  - Gauge charts for match scores
  - Pie charts for skill distribution
  - Bar charts for skill comparison
- **Multiple Export Formats**: Download results as PDF, TXT, or CSV
- **Beautiful UI**: Modern gradient design with responsive layout
- **Real-time Analysis**: Instant results with live visualizations

## 🏗️ Project Structure

```
resume_screener/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
└── (Generated at runtime)
    ├── models/           # Cached BERT models
    └── exports/          # Exported analysis reports
```

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (for BERT model)
- Internet connection (first run to download model)

## 🚀 Installation

1. **Clone or download the project**
   ```bash
   cd resume_screener
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`

3. **Upload and analyze**
   - Upload a resume (PDF, DOCX, or TXT)
   - Provide job description (upload file or paste text)
   - Click "Analyze Resume"
   - View interactive results and visualizations

4. **Export results**
   - Download analysis as PDF, TXT, or CSV
   - Share reports with stakeholders

## 🎨 UI Features

### Color Scheme
- **Primary Gradient**: Purple to violet (`#667eea` → `#764ba2`)
- **Accent Colors**: 
  - Pink gradient for metrics
  - Blue gradient for skills
  - Responsive cards with glassmorphism effects

### Visualizations
1. **Gauge Chart**: Shows overall match percentage with color-coded zones
2. **Pie Chart**: Displays skill match distribution
3. **Bar Chart**: Compares matched vs missing skills
4. **Metric Cards**: Quick overview of key statistics

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **NLP Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Visualizations**: Plotly
- **Document Processing**: 
  - PyPDF2 (PDF files)
  - python-docx (Word files)
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Export**: FPDF

## 📊 Analysis Metrics

### Match Score Interpretation
- **75-100%**: Highly Recommended (Green)
- **60-74%**: Recommended (Yellow)
- **45-59%**: Maybe (Orange)
- **0-44%**: Not Recommended (Red)

### Extracted Information
- Semantic similarity score
- Matched skills
- Missing skills
- Top keywords (TF-IDF based)
- Overall recommendation

## 🎯 Use Cases

- **Recruitment Agencies**: Screen large volumes of resumes efficiently
- **HR Departments**: Pre-filter candidates before manual review
- **Job Seekers**: Optimize resume for specific job descriptions
- **Career Coaches**: Provide data-driven resume improvement advice

## 🔐 Privacy & Security

- All processing is done locally on your machine
- No data is sent to external servers (except model download)
- Files are not stored permanently
- Session-based analysis

## 🐛 Troubleshooting

### Model Download Issues
If the BERT model fails to download:
```bash
# Manually download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Memory Issues
If you encounter memory errors:
- Close other applications
- Use a smaller BERT model: `paraphrase-MiniLM-L3-v2`
- Increase system swap space

### PDF Extraction Issues
Some PDFs may have extraction issues:
- Use DOCX or TXT format instead
- Ensure PDF is not image-based (use OCR if needed)

## 🚧 Future Enhancements

- [ ] Multi-language support
- [ ] Batch resume processing
- [ ] Custom skill database
- [ ] ATS keyword optimization
- [ ] Interview question generation
- [ ] Candidate ranking system
- [ ] Integration with job boards
- [ ] Email notification system

## 📝 License

This project is open-source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Support

For issues and questions, please create an issue in the repository.

## 🙏 Acknowledgments

- Sentence Transformers library
- Streamlit framework
- Plotly visualization library
- The open-source community

---

**Made with ❤️ using AI and Python**
