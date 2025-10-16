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


__all__ = ["GeminiEmbeddings"]


class GeminiEmbeddings(Embeddings):
    """Generate text embeddings using Google Generative AI (Gemini) models.

    The class wraps `langchain_google_genai.GoogleGenerativeAIEmbeddings` and adds:
    1. Exponential-backoff retry on rate-limit errors (`ResourceExhausted`).
    2. Simple batching (default 96 – service hard limit).
    3. Automatic credential check via ``GOOGLE_API_KEY`` environment variable.

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

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        """Embed a single query string."""
        return self._impl.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        """Embed multiple texts in batches with retry and progress bar.

        Google API hard-limits batch size. A delay is added between batches to
        respect free-tier rate limits. If a hard rate limit is hit, it waits
        for 60s.
        """
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @backoff.on_exception(backoff.expo, ResourceExhausted, max_time=60)
    def _retry_embed(self, batch: List[str]) -> List[List[float]]:
        """Embed a single batch, retrying on 429/ResourceExhausted errors."""
        return self._impl.embed_documents(batch)
