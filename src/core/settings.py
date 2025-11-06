from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration for Multi-Document Analysis System.

    Environment variables are automatically mapped from SCREAMING_SNAKE_CASE field names.
    LLM-related fields are optional (only needed for Day 3 answer generation).
    """

    # Required for embeddings & retrieval (Day 1-2)
    google_api_key: str
    embedding_model: str = "models/gemini-embedding-001"
    
    # Caching settings
    cache_enabled: bool = True
    cache_dir: str = "src/.cache"

    # Reranker settings
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3

    # Optional LLM settings (needed for Day 3)
    llm_model: str = "gemini-2.0-flash-lite"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_top_p: float = 1.0
    llm_frequency_penalty: float = 0.0
    llm_presence_penalty: float = 0.0

    model_config = SettingsConfigDict(
        env_file="./.env",
        extra="ignore",
        env_file_encoding="utf-8",
        use_enum_values=True,
    )


settings = Settings()