from __future__ import annotations

"""Retriever implementation backed by Persistable FAISS index."""

import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.embeddings import GeminiEmbeddings
from src.vector_store import PersistableFAISS
from .base_retriever import BaseRetriever

__all__ = ["FaissRetriever"]


class FaissRetriever(BaseRetriever):
    """Concrete retriever that uses a FAISS vector store on disk."""

    def __init__(
        self,
        k: int = 5,
        index_dir: str | Path = "vector_store/pdf_rag",
        embeddings: Embeddings | None = None,
        documents: List[Document] | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embeddings = embeddings or GeminiEmbeddings()
        self.documents = documents  # If None, expects index already on disk
        super().__init__(k=k, embeddings=self.embeddings)

    # ------------------------------------------------------------------
    def init_vector_store(self, embeddings: Embeddings, **kwargs) -> VectorStore:  # type: ignore[override]
        """Load existing index or create one from supplied documents."""
        store_wrapper = PersistableFAISS(
            index_dir=self.index_dir,
            documents=self.documents if self.documents else [],
            embeddings=embeddings,
        )
        return store_wrapper.vector_store
