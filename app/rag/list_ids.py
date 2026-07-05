"""
Utility to list and search for IDs in the vector database.
This helps discover what IDs are available for filtering or direct lookup.
"""

import os
from typing import List, Optional, Dict, Any

from app.rag.loaders import load_documents_from_manifest, iter_manifest_sources
from app.rag.chunkers import chunk_document, ChunkConfig
from app.rag.vectorstores import get_vector_store


def get_source_ids_from_manifest() -> List[Dict[str, Any]]:
    """
    Get all source IDs defined in the manifest without loading documents.
    
    Returns:
        List of source info dictionaries with id, doc_type, etc.
    """
    sources = []
    for source in iter_manifest_sources():
        sources.append({
            "id": source["id"],
            "doc_type": source["doc_type"],
            "insurance_type": source["insurance_type"],
            "product_code": source["product_code"],
            "product_name": source["product_name"],
            "claim_type": source["claim_type"],
        })
    return sources


def get_expected_chunk_ids(
    use_semantic_chunking: bool = True,
    chunk_config: Optional[ChunkConfig] = None,
) -> List[str]:
    """
    Get the expected chunk IDs that would be created during ingestion.
    This is useful for knowing what IDs to search for without actually ingesting.
    
    Args:
        use_semantic_chunking: Whether to use semantic chunking.
        chunk_config: Optional chunk configuration.
    
    Returns:
        List of expected chunk IDs in format {source_id}_{chunk_index}.
    """
    if chunk_config is None:
        chunk_config = ChunkConfig()
    
    documents = load_documents_from_manifest()
    all_ids = []
    
    for doc in documents:
        chunks = chunk_document(doc, chunk_config, use_semantic=use_semantic_chunking)
        for chunk in chunks:
            all_ids.append(f"{chunk.source_id}_{chunk.chunk_index}")
    
    return all_ids


def list_ids_in_store(
    vector_backend: Optional[str] = None,
) -> List[str]:
    """
    List all IDs currently stored in the vector database.
    
    Note: This requires the store to have been previously ingested.
    For FAISS, this returns IDs from the in-memory store.
    For Chroma, this queries the collection.
    For Pinecone, this would require listing all vectors (not directly supported).
    
    Args:
        vector_backend: Vector store backend ("faiss", "chroma", "pinecone").
    
    Returns:
        List of chunk IDs in the store.
    """
    if vector_backend is None:
        vector_backend = os.getenv("VECTOR_BACKEND", "faiss")
    
    store = get_vector_store(backend=vector_backend)
    
    # FAISS store keeps track of IDs
    if hasattr(store, "_chunk_ids"):
        return store._chunk_ids
    
    # Chroma store can query for all IDs
    if hasattr(store, "_collection") and store._collection is not None:
        all_data = store._collection.get()
        return all_data.get("ids", [])
    
    # Pinecone doesn't have a direct list method
    if vector_backend == "pinecone":
        print("Warning: Pinecone doesn't support listing all IDs directly.")
        print("Use metadata filters to narrow down searches instead.")
        return []
    
    return []


def search_by_metadata_filter(
    query: str,
    query_embedding: List[float],
    filter: Dict[str, Any],
    k: int = 5,
    vector_backend: Optional[str] = None,
) -> List[Any]:
    """
    Search the vector database using metadata filters.
    This is the recommended way to find specific content.
    
    Args:
        query: The search query string.
        query_embedding: Embedding vector for the query.
        filter: Metadata filter (e.g., {"source_id": "health_policy_hdfcergo"}).
        k: Number of results to return.
        vector_backend: Vector store backend.
    
    Returns:
        List of (Chunk, score) tuples.
    """
    if vector_backend is None:
        vector_backend = os.getenv("VECTOR_BACKEND", "faiss")
    
    store = get_vector_store(backend=vector_backend)
    return store.search(query, query_embedding, k=k, filter=filter)


def print_id_summary():
    """
    Print a summary of available source IDs and expected chunk IDs.
    """
    print("\n" + "=" * 60)
    print("VECTOR DATABASE ID SUMMARY")
    print("=" * 60)
    
    # Get source IDs from manifest
    sources = get_source_ids_from_manifest()
    print(f"\nTotal sources in manifest: {len(sources)}")
    print("\nSource IDs by document type:")
    
    by_type = {}
    for source in sources:
        doc_type = source["doc_type"]
        if doc_type not in by_type:
            by_type[doc_type] = []
        by_type[doc_type].append(source["id"])
    
    for doc_type, ids in by_type.items():
        print(f"\n  {doc_type}:")
        for sid in ids:
            print(f"    - {sid}")
    
    # Get expected chunk IDs
    print("\n" + "-" * 60)
    print("Expected chunk IDs (after ingestion):")
    expected_ids = get_expected_chunk_ids()
    print(f"Total expected chunks: {len(expected_ids)}")
    
    # Group by source
    by_source = {}
    for cid in expected_ids:
        source = cid.rsplit("_", 1)[0]
        if source not in by_source:
            by_source[source] = 0
        by_source[source] += 1
    
    print("\nChunks per source:")
    for source, count in sorted(by_source.items()):
        print(f"  - {source}: {count} chunk(s)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="List IDs in vector database")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["faiss", "chroma", "pinecone"],
        default=None,
        help="Vector store backend (default: from env or faiss)",
    )
    parser.add_argument(
        "--list-stored",
        action="store_true",
        help="List IDs currently in the store (requires prior ingestion)",
    )
    parser.add_argument(
        "--list-expected",
        action="store_true",
        help="List expected IDs based on manifest (no ingestion required)",
    )
    
    args = parser.parse_args()
    
    if args.list_stored:
        ids = list_ids_in_store(args.backend)
        print(f"\nIDs in store ({args.backend or 'default'}):")
        for i, cid in enumerate(ids[:20]):  # Show first 20
            print(f"  {i+1}. {cid}")
        if len(ids) > 20:
            print(f"  ... and {len(ids) - 20} more")
    else:
        print_id_summary()