"""Script to generate text chunks from PDFs and save them to a JSON file."""

import logging
from pathlib import Path

from src.document_processor import DocumentProcessor

# --- Configuration ---
PDF_DIRECTORY = Path("data/sample_pdfs")
OUTPUT_JSON_PATH = Path("data/chunks.json")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Process PDFs and save the chunks to a JSON file."""
    logger.info("--- Starting Chunk Generation ---")

    # 1. Process all PDFs in the source directory
    logger.info(f"1. Processing PDFs from: {PDF_DIRECTORY.resolve()}")
    if not PDF_DIRECTORY.exists():
        logger.error(f"Directory not found: {PDF_DIRECTORY}")
        return

    # This function processes PDFs and saves the chunks to the specified file.
    chunk_objects = DocumentProcessor.process_pdfs_in_directory(
        PDF_DIRECTORY,
        output_json=OUTPUT_JSON_PATH
    )

    if not chunk_objects:
        logger.warning("No documents were processed.")
        return

    logger.info(f"\n--- ✅ Chunk Generation Complete! ---")
    logger.info(f"{len(chunk_objects)} chunks saved to: {OUTPUT_JSON_PATH.resolve()}")


if __name__ == "__main__":
    main()