"""Script to build a FAISS vector store from PDFs in a directory."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from src.document_processor import DocumentProcessor
from src.rag import RAGPipeline
from tqdm import tqdm

# --- Configuration ---
PDF_DIRECTORY = Path("data/sample_pdfs")
INDEX_DIRECTORY = Path("src/.index/main")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Process PDFs and build the FAISS index."""
    logger.info("--- Starting Index Build ---")

    # 1. Load environment variables from .env file
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found. Please create a .env file.")
        return

    # 2. Process all PDFs in the source directory
    logger.info(f"1. Processing PDFs from: {PDF_DIRECTORY.resolve()}")
    if not PDF_DIRECTORY.exists():
        logger.error(f"Directory not found: {PDF_DIRECTORY}")
        return

    # Use the batch processor to get chunks from all PDFs
    # The title for each doc will be its filename
    chunk_objects = DocumentProcessor.process_pdfs_in_directory(PDF_DIRECTORY)
    if not chunk_objects:
        logger.warning("No documents were processed. Exiting.")
        return
    logger.info(f"   Generated {len(chunk_objects)} chunks.")

    # 3. Convert chunks to LangChain Document format
    logger.info("2. Converting chunks to LangChain Document format...")
    documents = [
        Document(
            page_content=chunk.content,
            metadata={
                "title": chunk.metadata.get("title", "Unknown Title"),
                "page": str(chunk.metadata.get("page_num", 0)),
            },
        )
        for chunk in tqdm(chunk_objects, desc="Converting Chunks")
    ]

    # 4. Initialize RAG pipeline and ingest documents
    logger.info(f"3. Building FAISS index at: {INDEX_DIRECTORY.resolve()}")
    pipeline = RAGPipeline(index_dir=INDEX_DIRECTORY)
    pipeline.ingest(documents)

    logger.info("\n--- ✅ Index Build Complete! ---")
    logger.info(f"Vector store created at: {INDEX_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()