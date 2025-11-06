from __future__ import annotations

"""Reranker implementation using cross-encoder models for refining search results."""

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

__all__ = ["Reranker"]


class Reranker:
    """Reranks documents using a cross-encoder model for improved relevance.
    
    Cross-encoders are more powerful than bi-encoders (used in initial retrieval)
    but slower, making them ideal for reranking a smaller set of candidates.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the reranker with a cross-encoder model.
        
        Parameters
        ----------
        model_name
            Name of the cross-encoder model (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").
        """
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Rerank documents based on query relevance.
        
        Parameters
        ----------
        query
            The search query string.
        documents
            List of documents to rerank.
            
        Returns
        -------
        List[Tuple[Document, float]]
            Documents sorted by relevance score (highest first), each with its score.
        """
        if not documents:
            return []
        
        # Create query-document pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores from the cross-encoder
        scores = self.model.predict(pairs)
        
        # Combine documents with scores and sort by score (descending)
        doc_with_scores = list(zip(documents, scores))
        doc_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_with_scores

