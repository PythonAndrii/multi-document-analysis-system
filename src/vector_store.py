from __future__ import annotations

"""Persistable FAISS vector store helper.

Wraps LangChain's FAISS wrapper to provide load-or-create convenience and
metadata persistence in a sibling pickle file.
"""

from pathlib import Path
import pickle
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS

__all__ = ["PersistableFAISS"]


class PersistableFAISS:
    """Load or create a FAISS index + metadata sidecar.

    Parameters
    ----------
    index_dir:
        Directory where FAISS files (and meta.pkl) live.
    documents:
        Collection of `Document` objects used when building a fresh index.
    embeddings:
        Embeddings provider; must match saved index on reload.
    """

    META_FILENAME = "meta.pkl"

    def __init__(
        self,
        index_dir: str | Path,
        documents: Sequence[Document] | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.index_dir / self.META_FILENAME

        if embeddings is None:
            raise ValueError("Embeddings instance must be supplied to PersistableFAISS")
        self.embeddings = embeddings

        # Either load existing or build new index
        if any(self.index_dir.iterdir()):
            self.store = self._load()
        else:
            if documents is None:
                raise ValueError("No existing index and no documents provided to build one.")
            self.store = self._build(list(documents))

    # ------------------------------------------------------------------
    @property
    def vector_store(self) -> VectorStore:  # convenience alias
        return self.store

    # ------------------------------------------------------------------
    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Proxy to underlying LangChain FAISS similarity search."""
        return self.store.similarity_search_with_score(query, k=k)

    # ------------------------------------------------------------------
    def _load(self) -> FAISS:
        store: FAISS = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        # load metadata list
        if self._meta_path.exists():
            with open(self._meta_path, "rb") as f:
                self.metadata: List[dict] = pickle.load(f)
        else:
            self.metadata = []
        return store

    # ------------------------------------------------------------------
    def _build(self, documents: List[Document]) -> FAISS:
        # Build new FAISS from documents
        store: FAISS = FAISS.from_documents(documents, self.embeddings)
        # Save index & metadata
        store.save_local(str(self.index_dir))
        self.metadata = [d.metadata for d in documents]
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        return store
