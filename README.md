# 📚 PYQ Analyzer — Previous Year Question Paper Analyzer

An AI-powered tool that analyzes VTU engineering question papers to identify patterns, important topics, and generate practice questions with answers.

## 🚀 Features
- Analyzes previous year question papers across multiple subjects
- Identifies frequently asked topics and repeating questions
- Combines question papers with study notes for comprehensive answers
- Supports uploading custom PDFs directly from the UI
- Subject-wise filtering for targeted exam preparation

## 🛠️ Tech Stack
- **RAG Pipeline** — LangChain + ChromaDB
- **Embeddings** — HuggingFace (all-MiniLM-L6-v2) — runs locally, no API needed
- **LLM** — Groq (llama-3.3-70b-versatile)
- **UI** — Streamlit
- **Document Loading** — PyPDF

## 📂 Subjects Covered
- Operating Systems (OS)
- Database Management Systems (DBMS)
- Computer Networks (CN)
- Data Structures & Algorithms (DSA)
- Artificial Intelligence & Machine Learning (AIML)

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/poorvi-16/pyq-analyzer.git
cd pyq-analyzer
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install langchain langchain-community langchain-chroma langchain-groq langchain-google-genai chromadb pypdf streamlit python-dotenv sentence-transformers langchain-text-splitters
```

### 4. Set up API keys
Create a `.env` file:

### 5. Add your PDFs
data/
├── OS/
│   ├── OS_qp1.pdf      # question papers
│   └── OS_notes1.pdf   # notes
├── DBMS/
└── ...


### 6. Run ingestion
```bash
python ingest.py
```

### 7. Launch the app
```bash
streamlit run app.py
```

## 🎯 How to Use
1. Select a subject from the sidebar
2. Ask questions like:
   - *"What are the most frequently asked topics?"*
   - *"Generate 5 practice questions with answers"*
   - *"Which units are most important for exams?"*
3. Upload your own PDFs directly from the sidebar

## 👩‍💻 Built By
Poorvika Srinivas — 3rd Year CS Student, SJBIT