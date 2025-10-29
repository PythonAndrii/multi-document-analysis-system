from __future__ import annotations

"""Document Processor Module

This module provides a small, SOLID-oriented component for turning a PDF into
overlapping text chunks with rich metadata suitable for vector‐store ingestion.
It purposefully keeps responsibilities separated:

1. PdfLoader               – Load raw page text from a PDF file.
2. PageChunker              – Split pages into chunks.
3. DocumentProcessor        – Orchestrator façade tying the two together.

All public classes depend only on abstractions (simple dataclasses / protocols)
so that underlying libraries (PyMuPDF, LangChain) can be swapped without
impacting callers.  Keeps configuration injectable via a single dataclass.
"""

import json
from typing import Dict, List
from pathlib import Path

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.dto.data_processing import Page, Chunk, DocumentProcessorConfig
from src.utils.helpers import write_chunks_json
from src.utils.loggers import logger
from tqdm import tqdm
import re


# ---------------------------------------------------------------------------
# PdfLoader – single responsibility: read PDF pages
# ---------------------------------------------------------------------------


class PdfLoader:  # noqa: D101
    """Load text of each page from a PDF."""

    def load(self, pdf_path: str | Path) -> List[Page]:  # noqa: D401 – imperative
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: List[Page] = []
        try:
            with fitz.open(path) as doc:  # type: ignore[arg-type]
                for idx, page in enumerate(doc):
                    raw_text = page.get_text("text")
                    text = self._clean_text(raw_text)
                    pages.append(Page(page_num=idx + 1, text=text))
        except Exception as exc:  # pragma: no cover – unexpected lib errors
            logger.exception("Failed to read PDF '%s': %s", pdf_path, exc)
            raise
        return pages

    def _clean_text(self, text: str) -> str:  # noqa: D401 – internal helper
        """Normalize PDF text to avoid word splits.

        - Removes hyphenation at line breaks (e.g. "liter-\nature" -> "literature")
        - Strips excess whitespace while preserving double newlines.
        """
        # Remove hyphen followed by linebreak(s)
        text = re.sub(r"-\s*\n\s*", "", text)
        # Collapse single newlines into spaces but keep paragraph breaks (two+ newlines)
        text = text.replace("\r", "")
        text = re.sub(r"\n{2,}", "<P>", text)  # temp marker for paragraphs
        text = text.replace("\n", " ")
        text = text.replace("<P>", "\n\n")
        # Normalize multiple spaces
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()


# ---------------------------------------------------------------------------
# PageChunker – single responsibility: split pages into chunks
# ---------------------------------------------------------------------------


class PageChunker:  # noqa: D101
    """Split :class:`Page` objects into overlapping text chunks."""

    def __init__(self, config: DocumentProcessorConfig):
        self._config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=[". ", "! ", "? ", "; ", ": ", " ", ""],
            keep_separator=False,
        )

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def chunk_pages(self, pages: List[Page]) -> List[Chunk]:
        """Return a flat list of chunks across all pages."""
        chunks: List[Chunk] = []

        for page in pages:
            page_chunks = self._splitter.split_text(page.text)
            for local_idx, chunk_text in enumerate(page_chunks, start=1):
                chunk_id = f"{self._config.title}-p{page.page_num}"
                metadata = {
                    "title": self._config.title,
                    "page_num": str(page.page_num),
                    "source": self._config.source,
                }
                chunks.append(Chunk(id=chunk_id, content=chunk_text, metadata=metadata))

        return chunks


# ---------------------------------------------------------------------------
# DocumentProcessor – façade orchestrating loader + chunker
# ---------------------------------------------------------------------------


class DocumentProcessor:  # noqa: D101
    """High-level API: PDF path in → List[Chunk] out."""

    def __init__(self, config: DocumentProcessorConfig | None = None):
        self._config = config or DocumentProcessorConfig()
        self._loader = PdfLoader()
        self._chunker = PageChunker(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pdf_path: str | Path) -> List[Chunk]:  # noqa: D401 – imperative
        """Convert *pdf_path* into text chunks with metadata."""
        self._config.source = str(pdf_path)  # capture absolute source path
        pages = self._loader.load(pdf_path)
        return self._chunker.chunk_pages(pages)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def config(self) -> DocumentProcessorConfig:  # noqa: D401 – simple prop
        return self._config

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def process_pdfs_in_directory(
        directory: str | Path,
        *,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        output_json: str | Path | None = "data/chunks.json",
    ) -> List[Chunk]:
        """Process every *.pdf* in *directory* and optionally persist chunks.

        Each PDF is processed with its filename stem as *title*.  A unique
        running ``global_id`` is added to the metadata so every chunk can be
        distinguished across multiple documents.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)

        combined: List[Chunk] = []
        global_id = 0

        for pdf_path in tqdm(sorted(directory.glob("*.pdf")), desc="Processing PDFs"):
            title = pdf_path.stem
            cfg = DocumentProcessorConfig(
                title=title,
                source=str(pdf_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            processor = DocumentProcessor(cfg)
            chunks = processor.process(pdf_path)

            for chunk in chunks:
                chunk.metadata["global_id"] = str(global_id)
                combined.append(chunk)
                global_id += 1

        # Persist to JSON if requested
        if output_json:
            write_chunks_json(output_json, combined)
            logger.info("Wrote %s chunks → %s", len(combined), output_json)

        return combined
