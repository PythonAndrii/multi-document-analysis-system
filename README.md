# multi-document-analysis-system
An open-source agentic Retrieval-Augmented Generation (RAG) system that answers complex questions by synthesising information from **multiple PDF documents**.

## Key Features
1. **PDF ingestion & chunking** – converts text-based PDFs (<50 pages) into page-annotated chunks.
2. **Semantic search (FAISS + Gemini embeddings)** – fast, local vector store with metadata for citations.
3. **Agentic reasoning (LangChain)** – multi-tool agent (`search_docs`, `summarise`) powered by Gemini 2.0 Flash-Lite LLM.
4. **Inline citations** – references returned as `[Doc-pX-cY]` pointing to page & chunk.
5. **Evaluation harness** – automated metrics for answer relevance, citation accuracy & completeness.
6. **Extensible CLI & optional Streamlit UI** – query from terminal today, add UI later.

## Directory Layout (planned)
```
├── src/
│   ├── document_processor.py   # PDF -> chunks
│   ├── embeddings.py           # Gemini embedding wrapper
│   ├── retrieval.py            # FAISS persistence & retriever
│   ├── agent.py                # LangChain agent setup
│   └── tools.py                # LangChain tool definitions
├── data/
│   └── sample_pdfs/            # Google, Microsoft, AWS, OpenAI, FLI docs
├── tests/                      # pytest suite
├── results/                    # evaluation outputs
├── requirements.txt            # pinned deps (pip)
└── README.md
```
