from __future__ import annotations

"""High-level orchestration for Retrieval-Augmented Generation pipeline.

This module focuses on *retrieval* only; answer generation with an LLM will be
integrated in Day-3.
"""

from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.core.client import get_llm
from src.embeddings import GeminiEmbeddings
from src.retrievers.faiss_retriever import FaissRetriever
from src.utils.prompts import get_rag_prompt
from src.core.settings import settings
from src.utils.loggers import logger

__all__ = ["RAGPipeline"]


class RAGPipeline:
    """Builds a FAISS-backed retriever from pre-chunked Documents and answers queries."""

    def __init__(
        self,
        k: int = 5,
        index_dir: str | Path = "src/.index/main",
        embeddings: GeminiEmbeddings | None = None,
    ) -> None:
        self.k = k
        self.index_dir = Path(index_dir)
        self.embeddings = embeddings or GeminiEmbeddings()
        self.retriever: FaissRetriever | None = None
        self.llm = get_llm(model=settings.llm_model, temperature=settings.llm_temperature)

    # ------------------------------------------------------------------
    def ingest(self, chunks: Iterable[Document]) -> None:
        """Create (or update) the FAISS index from a collection of Document chunks."""
        docs: List[Document] = list(chunks)
        if not docs:
            raise ValueError("No documents supplied for ingestion.")

        # Build retriever (this writes index if it doesn't exist)
        self.retriever = FaissRetriever(
            k=self.k,
            index_dir=self.index_dir,
            embeddings=self.embeddings,
            documents=docs,
        )

    # ------------------------------------------------------------------
    def query(self, question: str) -> List[Tuple[str, str, float]]:
        """Return list of (chunk_text, citation, score) for the query."""
        if self.retriever is None:
            # Attempt to load existing index (no docs)
            self.retriever = FaissRetriever(
                k=self.k,
                index_dir=self.index_dir,
                embeddings=self.embeddings,
            )
        # Use the underlying vector store to access similarity scores
        results = self.retriever._vector_store.similarity_search_with_score(question, k=self.k)  # type: ignore[attr-defined]
        formatted: List[Tuple[str, str, float]] = []
        for doc, score in results:
            meta = doc.metadata or {}
            citation = f"[{meta.get('title', 'unknown')}-p{meta.get('page', '?')}]"
            formatted.append((doc.page_content, citation, score))
        return formatted

    def generate_answer(self, question: str) -> str:
        """Generate an answer to a question based on retrieved context."""
        retrieved_chunks = self.query(question)
        if not retrieved_chunks:
            return "I couldn't find any relevant information to answer your question."

        context = "\n\n---\n\n".join(
            f"\nDocument: {text}"
            for text, _, _ in retrieved_chunks
        )


        prompt = get_rag_prompt()
        rag_chain = prompt | self.llm | StrOutputParser()
        logger.info(f"Generating answer for question: {question}")
        logger.info(f"Context: {context}")
        logger.info(f"LLM: {self.llm}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"RAG Chain: {rag_chain}")
        return rag_chain.invoke({"question": question, "context": context})
