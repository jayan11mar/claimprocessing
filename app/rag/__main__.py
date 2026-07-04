"""
CLI entry point for knowledge base ingestion.
Run with: python -m app.rag.ingest_basic
"""

import os
import sys
from typing import Optional

from app.rag.retriever_basic import get_retriever_with_stats, print_ingestion_summary


def main(embedding_model: Optional[str] = None, vector_backend: Optional[str] = None) -> None:
    """
    Main entry point for knowledge base ingestion.

    Args:
        embedding_model: Name of the embedding model to use.
        vector_backend: Vector store backend ("faiss", "chroma", "pinecone").
    """
    # Get configuration from environment or arguments
    if vector_backend is None:
        vector_backend = os.getenv("VECTOR_BACKEND", "faiss")

    if embedding_model is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    print(f"Starting knowledge base ingestion...")
    print(f"Vector backend: {vector_backend}")
    print(f"Embedding model: {embedding_model}")

    try:
        result = get_retriever_with_stats(
            embedding_model=embedding_model,
            vector_backend=vector_backend,
        )

        print_ingestion_summary(result["stats"])

        print(f"\nIngestion complete! Store contains {result['store'].chunk_count} chunks.")

    except FileNotFoundError as e:
        print(f"Error: Knowledge base file not found: {e}")
        print("Please ensure the knowledge base files are in place.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest knowledge base documents")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--vector-backend",
        type=str,
        choices=["faiss", "chroma", "pinecone"],
        default=None,
        help="Vector store backend (default: faiss)",
    )

    args = parser.parse_args()
    main(embedding_model=args.embedding_model, vector_backend=args.vector_backend)