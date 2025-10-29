"""File-based caching for embeddings using Python's shelve module."""

import shelve
from pathlib import Path
from typing import List, Optional

__all__ = ["EmbeddingCache"]


class EmbeddingCache:
    """File-based cache for storing and retrieving text embeddings.
    
    Uses Python's `shelve` module for persistent key-value storage.
    Keys are text strings, values are embedding vectors (lists of floats).
    """

    def __init__(self, cache_dir: str | Path) -> None:
        """Initialize the embedding cache.
        
        Parameters
        ----------
        cache_dir
            Directory where the cache file will be stored.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = str(self.cache_dir / "embeddings_cache")

    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve an embedding from the cache.
        
        Parameters
        ----------
        text
            The text string to look up.
            
        Returns
        -------
        Optional[List[float]]
            The cached embedding if found, None otherwise.
        """
        try:
            with shelve.open(self.cache_file) as cache:
                return cache.get(text)
        except Exception:
            # If cache file is corrupted or inaccessible, return None
            return None

    def set(self, text: str, embedding: List[float]) -> None:
        """Store an embedding in the cache.
        
        Parameters
        ----------
        text
            The text string to use as the key.
        embedding
            The embedding vector to store.
        """
        try:
            with shelve.open(self.cache_file) as cache:
                cache[text] = embedding
        except Exception:
            # If cache write fails, silently continue (don't break embedding flow)
            pass

