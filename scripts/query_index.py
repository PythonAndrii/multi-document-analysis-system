"""Script to query the pre-built FAISS vector store."""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.rag import RAGPipeline

# --- Configuration ---
INDEX_DIRECTORY = Path("src/.index/main")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Load the index and perform a query from the command line."""
    # 1. Load environment variables
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found. Please create a .env file.")
        return

    # 2. Check for query argument
    if len(sys.argv) < 2:
        logger.warning('Usage: python scripts/query_index.py "Your question here"')
        return
    question = sys.argv[1]

    # 3. Check if the index exists
    if not INDEX_DIRECTORY.exists() or not any(INDEX_DIRECTORY.iterdir()):
        logger.error(f"Index not found at '{INDEX_DIRECTORY.resolve()}'.")
        logger.error("Please run 'python scripts/create_index.py' first.")
        return

    # 4. Initialize the RAG pipeline
    # The pipeline will load the existing index since the directory is not empty
    logger.info(f"Loading index from: {INDEX_DIRECTORY.resolve()}...")
    pipeline = RAGPipeline(index_dir=INDEX_DIRECTORY, k=5)

    # 5. Perform the query
    logger.info(f"Querying for: \"{question}\"")
    logger.info("-" * 50)
    try:
        results = pipeline.query(question)

        if not results:
            logger.warning("No relevant documents found.")
            return

        # 6. Print the results
        for i, (text, citation, score) in enumerate(results, 1):
            logger.info(f"[{i}] Citation: {citation} (Similarity Score: {score:.4f})")
            # Indent the text for readability
            indented_text = "\n".join("    " + line for line in text.splitlines())
            logger.info(indented_text)
            logger.info("-" * 20)

    except Exception as e:
        logger.error(f"An error occurred during the query: {e}", exc_info=True)


if __name__ == "__main__":
    main()