from __future__ import annotations

"""Common abstraction for retrievers built on top of LangChain vector stores."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

__all__ = ["BaseRetriever"]


class BaseRetriever(ABC):
    """Abstract base retriever—wraps a VectorStoreRetriever with a uniform API."""

    def __init__(
        self,
        k: int,
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> None:
        self._vector_store: VectorStore = self.init_vector_store(embeddings=embeddings, **kwargs)
        self._retriever: VectorStoreRetriever = self._vector_store.as_retriever(search_kwargs={"k": k})

    # ------------------------------------------------------------------
    @property
    def retriever(self) -> VectorStoreRetriever:
        """Access the underlying LangChain retriever."""
        if not self._retriever:  # pragma: no cover – safety
            raise ValueError("VectorStoreRetriever is not initialized.")
        return self._retriever

    # Convenient alias
    def invoke(self, query: str):  # type: ignore[override]
        return self.retriever.invoke(query)

    # ------------------------------------------------------------------
    @abstractmethod
    def init_vector_store(self, embeddings: Embeddings, **kwargs: Any) -> VectorStore:
        """Initialize and return the VectorStore (e.g., FAISS, Chroma, etc.)."""
