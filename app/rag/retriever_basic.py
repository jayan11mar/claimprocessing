"""
Basic retriever builder for the claims knowledge base.
Builds a complete retrieval pipeline from manifest to vector store.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.rag.loaders import load_documents_from_manifest
from app.rag.chunkers import chunk_document, ChunkConfig, Chunk
from app.rag.embeddings import get_embedding_fn
from app.rag.vectorstores import get_vector_store


def build_basic_retriever(
    embedding_model: Optional[str] = None,
    vector_backend: Optional[str] = None,
    chunk_config: Optional[ChunkConfig] = None,
    use_semantic_chunking: bool = True,
    metadata_filter: Optional[Dict[str, Any]] = None,
    filter: Optional[Dict[str, Any]] = None,
    search_kwargs: Optional[Dict[str, Any]] = None,
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
    settings = get_settings()
    if chunk_config is None:
        chunk_config = ChunkConfig(
            chunk_size=int(getattr(settings, "CHUNK_SIZE", 800)),
            chunk_overlap=int(getattr(settings, "CHUNK_OVERLAP", 100)),
        )

    effective_use_semantic = use_semantic_chunking
    chunk_strategy = str(getattr(settings, "CHUNKING_STRATEGY", "recursive")).lower()
    if chunk_strategy in {"recursive", "recursive_chunking"}:
        effective_use_semantic = False
    elif chunk_strategy in {"semantic", "semantic_chunking", "semantic-chunking"}:
        effective_use_semantic = True

    all_chunks: List[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_config, use_semantic=effective_use_semantic)
        all_chunks.extend(chunks)

    # Get embedding function
    embed_fn = get_embedding_fn(embedding_model)

    # Embed all chunks
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embed_fn(texts)

    # Get embedding dimension from the first embedding
    dimension = len(embeddings[0]) if embeddings else 1536

    # Get vector store
    store = get_vector_store(backend=vector_backend, dimension=dimension)

    # Clear any existing index data before adding fresh chunks.
    # FAISSStore.__init__ auto-loads persisted data from disk, and add()
    # appends to the existing index. Without this clear, old dummy-chunk
    # data accumulates across repeated ingestion runs.
    store.delete(ids=None)

    # Upsert chunks
    store.add(all_chunks, embeddings)

    # Persist
    store.persist()

    # Pass embedding_fn so the retriever can embed queries at search time
    effective_filter = metadata_filter if metadata_filter is not None else filter
    effective_search_kwargs = dict(search_kwargs or {})
    if "embedding_fn" not in effective_search_kwargs:
        effective_search_kwargs["embedding_fn"] = embed_fn
    if effective_filter is not None and "filter" not in effective_search_kwargs:
        effective_search_kwargs["filter"] = effective_filter
    if "k" not in effective_search_kwargs:
        effective_search_kwargs["k"] = 5

    return store.as_retriever(search_kwargs=effective_search_kwargs)


def get_retriever_with_stats(
    embedding_model: Optional[str] = None,
    vector_backend: Optional[str] = None,
    chunk_config: Optional[ChunkConfig] = None,
    use_semantic_chunking: bool = True,
    metadata_filter: Optional[Dict[str, Any]] = None,
    filter: Optional[Dict[str, Any]] = None,
    search_kwargs: Optional[Dict[str, Any]] = None,
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
    settings = get_settings()
    if chunk_config is None:
        chunk_config = ChunkConfig(
            chunk_size=int(getattr(settings, "CHUNK_SIZE", 800)),
            chunk_overlap=int(getattr(settings, "CHUNK_OVERLAP", 100)),
        )

    effective_use_semantic = use_semantic_chunking
    chunk_strategy = str(getattr(settings, "CHUNKING_STRATEGY", "recursive")).lower()
    if chunk_strategy in {"recursive", "recursive_chunking"}:
        effective_use_semantic = False
    elif chunk_strategy in {"semantic", "semantic_chunking", "semantic-chunking"}:
        effective_use_semantic = True

    all_chunks: List[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_config, use_semantic=effective_use_semantic)
        all_chunks.extend(chunks)
        stats["doc_type_counts"][doc.doc_type] += 1
        stats["chunks_by_doc_type"][doc.doc_type] += len(chunks)

    stats["total_chunks"] = len(all_chunks)

    # Get embedding function
    embed_fn = get_embedding_fn(embedding_model)

    # Embed all chunks
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embed_fn(texts)

    # Get embedding dimension from the first embedding
    dimension = len(embeddings[0]) if embeddings else 1536

    # Get vector store
    store = get_vector_store(backend=vector_backend, dimension=dimension)

    # Clear any existing index data before adding fresh chunks.
    store.delete(ids=None)

    # Upsert chunks
    store.add(all_chunks, embeddings)

    # Persist
    store.persist()

    # Pass embedding_fn so the retriever can embed queries at search time
    effective_filter = metadata_filter if metadata_filter is not None else filter
    effective_search_kwargs = dict(search_kwargs or {})
    if "embedding_fn" not in effective_search_kwargs:
        effective_search_kwargs["embedding_fn"] = embed_fn
    if effective_filter is not None and "filter" not in effective_search_kwargs:
        effective_search_kwargs["filter"] = effective_filter
    if "k" not in effective_search_kwargs:
        effective_search_kwargs["k"] = 5

    return {
        "retriever": store.as_retriever(search_kwargs=effective_search_kwargs),
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