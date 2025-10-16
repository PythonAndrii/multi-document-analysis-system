"""Script to ask a question to the pre-built FAISS vector store and get an answer."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

from src.rag import RAGPipeline

logger = logging.getLogger(__name__)

# --- Configuration ---
INDEX_DIRECTORY = Path("src/.index/main")


def main():
    """Load the index, perform a query, and generate an answer."""
    # 1. Load environment variables
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found. Please create a .env file.")
        return

    # 2. Check for query argument
    if len(sys.argv) < 2:
        logger.warning('Usage: python scripts/ask_question.py "Your question here"')
        return
    question = sys.argv[1]

    # 3. Check if the index exists
    if not INDEX_DIRECTORY.exists() or not any(INDEX_DIRECTORY.iterdir()):
        logger.error(f"Index not found at '{INDEX_DIRECTORY.resolve()}'.")
        logger.error("Please run 'python scripts/create_index.py' first.")
        return

    # 4. Initialize the RAG pipeline
    logger.info(f"Loading index from: {INDEX_DIRECTORY.resolve()}...")
    pipeline = RAGPipeline(index_dir=INDEX_DIRECTORY, k=5)

    # 5. Generate the answer
    logger.info(f"Generating answer for: \"{question}\"")
    logger.info("-" * 50)
    try:
        answer = pipeline.generate_answer(question)
        logger.info(answer)

    except Exception as e:
        logger.error(f"An error occurred during answer generation: {e}", exc_info=True)


if __name__ == "__main__":
    main()