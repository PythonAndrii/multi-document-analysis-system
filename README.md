# Multi-Document Analysis System

An open-source Retrieval-Augmented Generation (RAG) system that answers complex questions by synthesizing information from **multiple PDF documents** with inline citations.

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey))

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/multi-document-analysis-system.git
cd multi-document-analysis-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the project root:

```bash
# Required
GOOGLE_API_KEY=your_google_ai_studio_key_here
EMBEDDING_MODEL=models/gemini-embedding-001

# Optional
LLM_MODEL=gemini-2.0-flash-exp
```

---

## 🚀 Usage

### 1. Place PDFs in `data/sample_pdfs`
Create this directory if it doesn't exist and add the PDF files you want to analyze.

### 2. Create the Vector Index
This command processes the PDFs, generates embeddings, and builds a FAISS vector store.

```bash
python scripts/create_index.py
```

### 3. Query for Similar Chunks
Retrieve raw text chunks most relevant to your question.

```bash
python scripts/query_index.py "Your question here"
```

### 4. Ask a Question
Get a generated answer from the RAG pipeline.

```bash
python scripts/ask_question.py "Your question here"
```

---

## 📁 Project Structure

```
multi-document-analysis-system/
├── data/
│   └── sample_pdfs/             # Source documents
├── scripts/
│   ├── create_index.py          # Script to build the vector store
│   ├── query_index.py           # Script to query the index
│   └── ask_question.py          # Script to ask a question to the RAG pipeline
├── src/
│   ├── core/
│   │   ├── client.py              # Gemini API client
│   │   └── settings.py          # Pydantic settings
│   ├── retrievers/
│   │   ├── base_retriever.py    # Abstract retriever interface
│   │   └── faiss_retriever.py   # FAISS implementation
│   ├── utils/
│   │   ├── prompts.py           # RAG prompt templates
│   │   └── loggers.py           # Logging setup
│   ├── document_processor.py    # PDF → chunks pipeline
│   ├── embeddings.py            # Gemini embeddings wrapper
│   ├── vector_store.py          # FAISS persistence helper
│   └── rag.py                   # High-level RAG orchestration
├── tests/
│   ├── data/sample.pdf          # Test fixture
│   ├── test_document_processor.py
│   └── test_rag_pipeline.py
├── .env                         # Environment variables (not in git)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🧪 Testing

All tests use `DummyEmbeddings` to avoid hitting Google API quotas during development.

```bash
# Quick test - document processing
pytest tests/test_document_processor.py::test_document_processor_returns_metadata -v

# Quick test - RAG pipeline
pytest tests/test_rag_pipeline.py::test_rag_pipeline_ingest_and_query -v

# Full test suite with coverage
pytest tests/ --cov=src --cov-report=html
```


---

## 📝 Citation Format

Citations follow the pattern: `[DocumentTitle-pX]`
- `DocumentTitle`: The title from `DocumentProcessorConfig`
- `pX`: Page number (1-indexed)

Example: `[AIReport-p3]` → Page 3 of document titled "AIReport"
