"""
Basic retriever builder for the claims knowledge base.
Builds a complete retrieval pipeline from manifest to vector store.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from langchain_core.documents import Document as LangChainDocument

from app.rag.loaders import load_documents_from_manifest, Document
from app.rag.chunkers import chunk_document, ChunkConfig, Chunk
from app.rag.embeddings import get_embedding_fn
from app.rag.vectorstores import get_vector_store, VectorStore


def build_basic_retriever(
    embedding_model: Optional[str] = None,
    vector_backend: Optional[str] = None,
    chunk_config: Optional[ChunkConfig] = None,
    use_semantic_chunking: bool = True,
) -> Any:
    """
    Build a basic retriever over the claims knowledge base.

    This function:
    - Calls load_documents_from_manifest() to read all sources
    - Applies recursive/semantic chunking to each document
    - Embeds chunks using the chosen embedding model
    - Upserts chunks into the configured vector store
    - Returns a LangChain VectorStoreRetriever

    Args:
        embedding_model: Name of the embedding model to use.
        vector_backend: Vector store backend ("faiss", "chroma", "pinecone").
        chunk_config: Configuration for chunking.
        use_semantic_chunking: Whether to use semantic chunking for structured docs.

    Returns:
        A LangChain VectorStoreRetriever over the knowledge base.
    """
    # Get configuration
    if vector_backend is None:
        vector_backend = os.getenv("VECTOR_BACKEND", "faiss")

    # Load documents
    documents = load_documents_from_manifest()

    # Chunk documents
    if chunk_config is None:
        chunk_config = ChunkConfig()

    all_chunks: List[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_config, use_semantic=use_semantic_chunking)
        all_chunks.extend(chunks)

    # Get embedding function
    embed_fn = get_embedding_fn(embedding_model)

    # Embed all chunks
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embed_fn(texts)

    # Get vector store
    store = get_vector_store(backend=vector_backend)

    # Upsert chunks
    store.add(all_chunks, embeddings)

    # Persist
    store.persist()

    # Return retriever
    return store.as_retriever()


def get_retriever_with_stats(
    embedding_model: Optional[str] = None,
    vector_backend: Optional[str] = None,
    chunk_config: Optional[ChunkConfig] = None,
    use_semantic_chunking: bool = True,
) -> Dict[str, Any]:
    """
    Build a retriever and return statistics about the ingestion.

    Args:
        embedding_model: Name of the embedding model to use.
        vector_backend: Vector store backend ("faiss", "chroma", "pinecone").
        chunk_config: Configuration for chunking.
        use_semantic_chunking: Whether to use semantic chunking for structured docs.

    Returns:
        Dictionary with retriever and statistics.
    """
    if vector_backend is None:
        vector_backend = os.getenv("VECTOR_BACKEND", "faiss")

    # Load documents
    documents = load_documents_from_manifest()

    # Track statistics
    stats = {
        "doc_type_counts": defaultdict(int),
        "total_documents": len(documents),
        "total_chunks": 0,
        "chunks_by_doc_type": defaultdict(int),
    }

    # Chunk documents
    if chunk_config is None:
        chunk_config = ChunkConfig()

    all_chunks: List[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_config, use_semantic=use_semantic_chunking)
        all_chunks.extend(chunks)
        stats["doc_type_counts"][doc.doc_type] += 1
        stats["chunks_by_doc_type"][doc.doc_type] += len(chunks)

    stats["total_chunks"] = len(all_chunks)

    # Get embedding function
    embed_fn = get_embedding_fn(embedding_model)

    # Embed all chunks
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embed_fn(texts)

    # Get vector store
    store = get_vector_store(backend=vector_backend)

    # Upsert chunks
    store.add(all_chunks, embeddings)

    # Persist
    store.persist()

    # Return retriever and stats
    return {
        "retriever": store.as_retriever(),
        "stats": dict(stats),
        "store": store,
    }


def print_ingestion_summary(stats: Dict[str, Any]) -> None:
    """
    Print a summary of the ingestion process.

    Args:
        stats: Statistics dictionary from get_retriever_with_stats.
    """
    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE INGESTION SUMMARY")
    print("=" * 60)

    print(f"\nTotal documents loaded: {stats['total_documents']}")
    print(f"Total chunks created: {stats['total_chunks']}")

    print("\nDocuments by type:")
    for doc_type, count in stats["doc_type_counts"].items():
        print(f"  - {doc_type}: {count} document(s)")

    print("\nChunks by document type:")
    for doc_type, count in stats["chunks_by_doc_type"].items():
        print(f"  - {doc_type}: {count} chunk(s)")

    print("\n" + "=" * 60)