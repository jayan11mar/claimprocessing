"""
CLI entry point for knowledge base ingestion.
Run with: python -m app.rag.ingest_basic

Loads manifest sources, runs loaders → chunkers → embeddings → vector store upsert,
and prints a summary per doc_type with document counts and total chunk counts.
"""

import os
import sys
from typing import Optional

from app.rag.loaders import load_documents_from_manifest
from app.rag.chunkers import chunk_document, ChunkConfig
from app.rag.embeddings import get_embedding_fn
from app.rag.vectorstores import get_vector_store


def main(
    embedding_model: Optional[str] = None,
    vector_backend: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    use_semantic_chunking: bool = True,
) -> None:
    """
    Main entry point for knowledge base ingestion.

    Pipeline: loaders → chunkers → embeddings → vector store upsert.

    Args:
        embedding_model: Name of the embedding model to use.
        vector_backend: Vector store backend ("faiss", "chroma", "pinecone").
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        use_semantic_chunking: Whether to use semantic chunking for structured docs.
    """
    # Get configuration from environment or arguments
    if vector_backend is None:
        vector_backend = os.getenv("VECTOR_BACKEND", "faiss")

    if embedding_model is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    print("=" * 60)
    print("KNOWLEDGE BASE INGESTION")
    print("=" * 60)
    print(f"Vector backend : {vector_backend}")
    print(f"Embedding model: {embedding_model}")
    print(f"Chunk size     : {chunk_size}")
    print(f"Chunk overlap  : {chunk_overlap}")
    print(f"Semantic       : {use_semantic_chunking}")
    print()

    try:
        # ── Step 1: Load documents from manifest ──
        print("Step 1/4: Loading documents from manifest...")
        documents = load_documents_from_manifest()
        print(f"  → {len(documents)} document(s) loaded")
        print()

        # ── Step 2: Chunk documents ──
        print("Step 2/4: Chunking documents...")
        chunk_config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = []
        for doc in documents:
            chunks = chunk_document(doc, chunk_config, use_semantic=use_semantic_chunking)
            all_chunks.extend(chunks)
        print(f"  → {len(all_chunks)} chunk(s) created")
        print()

        # ── Step 3: Embed chunks ──
        print("Step 3/4: Generating embeddings...")
        embed_fn = get_embedding_fn(embedding_model)
        texts = [chunk.text for chunk in all_chunks]
        embeddings = embed_fn(texts)
        print(f"  → {len(embeddings)} embedding(s) generated (dim={len(embeddings[0]) if embeddings else 'N/A'})")
        print()

        # ── Step 4: Upsert into vector store ──
        print("Step 4/4: Upserting into vector store...")
        store = get_vector_store(backend=vector_backend)
        store.add(all_chunks, embeddings)
        store.persist()
        print(f"  → {store.chunk_count} chunk(s) stored in {vector_backend} backend")
        print()

        # ── Summary per doc_type ──
        print("=" * 60)
        print("INGESTION SUMMARY BY DOCUMENT TYPE")
        print("=" * 60)

        # Aggregate counts by doc_type
        doc_type_docs: dict = {}
        doc_type_chunks: dict = {}
        for doc in documents:
            dt = doc.doc_type
            doc_type_docs[dt] = doc_type_docs.get(dt, 0) + 1
        for chunk in all_chunks:
            dt = chunk.doc_type
            doc_type_chunks[dt] = doc_type_chunks.get(dt, 0) + 1

        all_types = sorted(set(list(doc_type_docs.keys()) + list(doc_type_chunks.keys())))
        total_docs = 0
        total_chunks = 0
        for dt in all_types:
            d_count = doc_type_docs.get(dt, 0)
            c_count = doc_type_chunks.get(dt, 0)
            total_docs += d_count
            total_chunks += c_count
            print(f"  {dt:25s}  {d_count:3d} document(s)  {c_count:5d} chunk(s)")

        print("-" * 60)
        print(f"  {'TOTAL':25s}  {total_docs:3d} document(s)  {total_chunks:5d} chunk(s)")
        print("=" * 60)
        print("Ingestion complete!")

    except FileNotFoundError as e:
        print(f"Error: Knowledge base file not found: {e}", file=sys.stderr)
        print("Please ensure the knowledge base files are in place.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest knowledge base documents into vector store"
    )
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Target chunk size in characters (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between consecutive chunks (default: 100)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic chunking (use recursive chunking for all doc types)",
    )

    args = parser.parse_args()
    main(
        embedding_model=args.embedding_model,
        vector_backend=args.vector_backend,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
    )