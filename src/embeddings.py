from __future__ import annotations

"""Gemini embeddings wrapper (Google AI Studio) compatible with LangChain Embeddings interface."""

import time
from typing import List

import backoff
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

from src.core.settings import settings
from src.caching import EmbeddingCache


__all__ = ["GeminiEmbeddings"]


class GeminiEmbeddings(Embeddings):
    """Generate text embeddings using Google Generative AI (Gemini) models.

    The class wraps `langchain_google_genai.GoogleGenerativeAIEmbeddings` and adds:
    1. Exponential-backoff retry on rate-limit errors (`ResourceExhausted`).
    2. Simple batching (default 96 – service hard limit).
    3. Automatic credential check via ``GOOGLE_API_KEY`` environment variable.
    4. File-based caching to avoid redundant API calls.

    The returned vectors are already L2-normalised; use FAISS `IndexFlatIP` for cosine similarity.
    """

    def __init__(
        self,
        task_type: str | None = "retrieval_document",
        batch_size: int = 32,  # Reduced batch size
    ) -> None:
        """Parameters
        ----------
        task_type
            Task hint for Google embeddings API.
        batch_size
            Max 96 (API hard-limit).
        """

        # Configure low-level client so downstream wrapper sees credentials
        genai.configure(api_key=settings.google_api_key)

        self.batch_size = batch_size
        self._impl = GoogleGenerativeAIEmbeddings(model=settings.embedding_model, task_type=task_type)
        self._cache = EmbeddingCache(settings.cache_dir) if settings.cache_enabled else None

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        """Embed a single query string."""
        return self._impl.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        """Embed multiple texts in batches with retry and progress bar.

        Uses caching if enabled to avoid redundant API calls.
        Google API hard-limits batch size. A delay is added between batches to
        respect free-tier rate limits. If a hard rate limit is hit, it waits
        for 60s.
        """
        if self._cache:
            return self._embed_with_cache(texts)
        return self._embed_without_cache(texts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed_with_cache(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using cache to avoid redundant API calls."""
        vectors: List[List[float]] = []
        texts_to_embed: List[str] = []
        text_indices: List[int] = []
        
        # Check cache for each text
        for idx, text in enumerate(texts):
            cached = self._cache.get(text) if self._cache else None
            if cached is not None:
                vectors.append(cached)
            else:
                vectors.append(None)  # Placeholder
                texts_to_embed.append(text)
                text_indices.append(idx)
        
        # If all texts were cached, return immediately
        if not texts_to_embed:
            return vectors
        
        # Embed uncached texts
        pbar = tqdm(total=len(texts), desc="Embedding documents", unit="docs")
        
        # Update progress bar for cached items
        cached_count = len(texts) - len(texts_to_embed)
        pbar.update(cached_count)
        
        # Process uncached texts in batches
        for i in range(0, len(texts_to_embed), self.batch_size):
            batch = texts_to_embed[i : i + self.batch_size]
            batch_indices = text_indices[i : i + self.batch_size]

            # Inner loop to handle hard rate limits with a long pause
            while True:
                try:
                    embedded_batch = self._retry_embed(batch)
                    # Store embeddings in cache and result vector
                    for text, embedding, idx in zip(batch, embedded_batch, batch_indices):
                        if self._cache:
                            self._cache.set(text, embedding)
                        vectors[idx] = embedding
                    pbar.update(len(batch))
                    break  # Success, exit retry loop for this batch
                except ResourceExhausted:
                    pbar.write(
                        "\nRate limit exceeded. Waiting 60 seconds before retrying..."
                    )
                    time.sleep(60)

            # If it's not the last batch, wait to avoid hitting rate limits for free tier TPM
            is_last_batch = (i + self.batch_size) >= len(texts_to_embed)
            if not is_last_batch:
                time.sleep(30)  # Increased 30-second delay between batches

        pbar.close()
        return vectors

    def _embed_without_cache(self, texts: List[str]) -> List[List[float]]:
        """Embed documents without caching (original implementation)."""
        vectors: List[List[float]] = []
        pbar = tqdm(total=len(texts), desc="Embedding documents", unit="docs")
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Inner loop to handle hard rate limits with a long pause
            while True:
                try:
                    embedded_batch = self._retry_embed(batch)
                    vectors.extend(embedded_batch)
                    pbar.update(len(batch))
                    break  # Success, exit retry loop for this batch
                except ResourceExhausted:
                    pbar.write(
                        "\nRate limit exceeded. Waiting 60 seconds before retrying..."
                    )
                    time.sleep(60)

            # If it's not the last batch, wait to avoid hitting rate limits for free tier TPM
            is_last_batch = (i + self.batch_size) >= len(texts)
            if not is_last_batch:
                time.sleep(30)  # Increased 30-second delay between batches

        pbar.close()
        return vectors

    @backoff.on_exception(backoff.expo, ResourceExhausted, max_time=60)
    def _retry_embed(self, batch: List[str]) -> List[List[float]]:
        """Embed a single batch, retrying on 429/ResourceExhausted errors."""
        return self._impl.embed_documents(batch)
