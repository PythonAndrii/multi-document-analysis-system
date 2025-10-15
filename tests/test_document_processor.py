import importlib
from pathlib import Path

import pytest
import json

from src.document_processor import (
    DocumentProcessor,
    DocumentProcessorConfig,
    PdfLoader,
    PageChunker,
)


TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
SAMPLE_PDF = DATA_DIR / "sample.pdf"


@pytest.fixture(scope="session")
def processor() -> DocumentProcessor:
    config = DocumentProcessorConfig(title="TestDoc")
    return DocumentProcessor(config)


def test_page_chunker_respects_chunk_size():
    loader = PdfLoader()
    pages = loader.load(SAMPLE_PDF)
    cfg = DocumentProcessorConfig(chunk_size=200, chunk_overlap=0, title="TestChunker")
    chunker = PageChunker(cfg)
    chunks = chunker.chunk_pages(pages)
    assert all(len(c.content) <= 200 for c in chunks)


def test_document_processor_returns_metadata(processor: DocumentProcessor):
    chunks = processor.process(SAMPLE_PDF)
    assert chunks, "No chunks returned"
    first = chunks[0]
    for key in ["title", "page_num", "source"]:
        assert key in first.metadata


def test_output_chunks_for_inspection(processor: DocumentProcessor):
    """Output chunks to see what the processor generates."""
    chunks = processor.process(SAMPLE_PDF)
    
    print(f"\n=== Generated {len(chunks)} chunks ===")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Content: '{chunk.content[:400]}{'...' if len(chunk.content) > 100 else ''}'")
        print(f"  Metadata: {chunk.metadata}")
    print("=" * 50)


def test_process_pdfs_in_directory(tmp_path: Path):
    """Ensure batch directory processing returns chunks and writes JSON."""
    # Arrange: copy sample PDF into a temporary directory
    sample_copy = tmp_path / "sample.pdf"
    sample_copy.write_bytes(SAMPLE_PDF.read_bytes())

    # Act: process directory and write output json
    output_json = tmp_path / "chunks.json"
    chunks = DocumentProcessor.process_pdfs_in_directory(
        tmp_path, output_json=output_json
    )

    # Assert: chunks list not empty and JSON file created
    assert chunks, "process_pdfs_in_directory returned no chunks"
    assert output_json.exists(), "Output JSON was not created"

    # Quick metadata sanity check on first chunk
    first_meta = chunks[0].metadata
    assert "global_id" in first_meta and first_meta["global_id"] == "0"

    # Load JSON and compare lengths
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(data) == len(chunks)
