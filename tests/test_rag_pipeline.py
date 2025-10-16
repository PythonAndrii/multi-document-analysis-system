import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Ensure pydantic Settings validation passes during tests by pre-seeding
# required environment variables with placeholder values.
# ---------------------------------------------------------------------------

os.environ.setdefault("GOOGLE_API_KEY", "dummy_key_for_tests")
os.environ.setdefault("EMBEDDING_MODEL", "models/embedding-001")

from src.rag import RAGPipeline
from src.document_processor import DocumentProcessor, DocumentProcessorConfig

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
SAMPLE_PDF = DATA_DIR / "sample.pdf"


class DummyEmbeddings(Embeddings):
    """Fast, deterministic embeddings for tests – avoids external API calls."""

    dim = 16  # small dimension for speed

    def _rand(self) -> List[float]:
        random.seed(42)  # deterministic across processes
        return [random.random() for _ in range(self.dim)]

    # ------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return [self._rand() for _ in texts]

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        return self._rand()


@pytest.fixture(scope="session")
def chunks() -> List[Document]:
    cfg = DocumentProcessorConfig(title="TestDoc")
    processor = DocumentProcessor(cfg)
    chunk_objs = processor.process(SAMPLE_PDF)

    docs: List[Document] = []
    for ch in chunk_objs:
        meta = {
            "title": ch.metadata["title"],
            "page": ch.metadata["page_num"],
        }
        docs.append(Document(page_content=ch.content, metadata=meta))
    return docs


def test_rag_pipeline_ingest_and_query(tmp_path: Path, chunks: List[Document]):
    """Full ingest→retrieve loop using DummyEmbeddings."""

    pipeline = RAGPipeline(
        k=3,
        index_dir=tmp_path / "faiss_index",
        embeddings=DummyEmbeddings(),
    )

    pipeline.ingest(chunks)

    # Use a snippet from the first chunk as the query
    query_snippet = chunks[0].page_content.split(" ")[:10]
    query = " ".join(query_snippet)

    results = pipeline.query(query)
    assert results, "No results returned"

    top_text, citation, score = results[0]
    assert citation.startswith("[TestDoc-p"), "Citation format incorrect"
    assert len(top_text) > 0
