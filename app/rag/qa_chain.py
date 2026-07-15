"""A lightweight QA chain that produces answer text and citations from hybrid retrieval."""

from typing import Any, Dict, List, Optional

from app.rag.chunkers import Chunk, ChunkConfig, chunk_document
from app.rag.loaders import load_documents_from_manifest
from app.rag.retriever_hybrid import hybrid_retrieve
from app.rag.vectorstores import get_vector_store


# ── Module-level persisted chunks cache ──────────────────────────────────────
_PERSISTED_CHUNKS: Optional[List[Chunk]] = None


def _load_chunks_from_manifest() -> List[Chunk]:
    documents = load_documents_from_manifest()
    chunks: List[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, ChunkConfig(chunk_size=800, chunk_overlap=100), use_semantic=True))
    return chunks


def _get_persisted_chunks() -> List[Chunk]:
    """
    Return chunks from the persisted vector store, falling back to building
    from manifest if the persisted index does not exist.

    The result is cached at module level so it is loaded only once.
    """
    global _PERSISTED_CHUNKS
    if _PERSISTED_CHUNKS is not None:
        return _PERSISTED_CHUNKS

    from app.config import get_settings
    from app.rag.vectorstores.faiss_store import FAISSStore

    persist_path = get_settings().VECTOR_PERSIST_PATH
    store = FAISSStore.load(persist_path)
    if store is not None and store.chunk_count > 0:
        _PERSISTED_CHUNKS = store.get_chunks()
    else:
        _PERSISTED_CHUNKS = _load_chunks_from_manifest()

    return _PERSISTED_CHUNKS


def _build_qa_payload(
    query: str,
    chunks: Optional[List[Chunk]] = None,
    claim_context: Optional[str] = None,
    top_k: int = 5,
    embedding_fn: Optional[Any] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if chunks is None:
        chunks = _get_persisted_chunks()

    if not chunks:
        return {
            "answer_text": "No knowledge-base content was available for this query.",
            "citations": [],
            "confidence": 0.0,
        }

    results = hybrid_retrieve(chunks, query, k=top_k, embedding_fn=embedding_fn, metadata_filter=metadata_filter)
    if not results:
        return {
            "answer_text": "No relevant guidance was found in the knowledge base.",
            "citations": [],
            "confidence": 0.0,
        }

    best_chunk = results[0]["chunk"]
    excerpt = best_chunk.text.strip().replace("\n", " ")
    if len(excerpt) > 260:
        excerpt = excerpt[:257] + "..."

    if claim_context:
        answer_text = f"For {claim_context}, the retrieved guidance says: {excerpt}"
    else:
        answer_text = f"The retrieved guidance says: {excerpt}"

    citations = []
    for result in results[: min(3, len(results))]:
        citations.append(
            {
                "chunk_id": result["chunk_id"],
                "text": result["chunk"].text,
                "source_id": result["source_id"],
                "source_path": result["source_path"],
                "score": result.get("rerank_score", result["combined_score"]),
            }
        )

    confidence = min(0.99, max(0.45, 0.55 + 0.08 * len(citations) + 0.03 * min(1.0, results[0]["combined_score"])))
    return {
        "answer_text": answer_text,
        "citations": citations,
        "confidence": round(confidence, 3),
    }


def run_qa_chain(
    query: str,
    chunks: Optional[List[Chunk]] = None,
    claim_context: Optional[str] = None,
    top_k: int = 5,
    embedding_fn: Optional[Any] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a hybrid retrieval QA flow and return answer text plus citations.

    Args:
        query: The query string.
        chunks: Optional list of Chunk objects. If None, loads from persisted index.
        claim_context: Optional claim context string for answer formatting.
        top_k: Number of results to return.
        embedding_fn: Optional embedding function for dense scoring.
        metadata_filter: Optional dict of metadata fields to filter chunks on.

    Returns:
        Dict with answer_text, citations, and confidence.
    """
    return _build_qa_payload(query, chunks=chunks, claim_context=claim_context, top_k=top_k, embedding_fn=embedding_fn, metadata_filter=metadata_filter)


def stream_qa_chain(
    query: str,
    chunks: Optional[List[Chunk]] = None,
    claim_context: Optional[str] = None,
    top_k: int = 5,
    embedding_fn: Optional[Any] = None,
    chunk_size: int = 24,
    metadata_filter: Optional[Dict[str, Any]] = None,
):
    """Yield a QA answer incrementally in small text chunks for streaming-style UIs."""
    payload = _build_qa_payload(query, chunks=chunks, claim_context=claim_context, top_k=top_k, embedding_fn=embedding_fn, metadata_filter=metadata_filter)
    answer_text = payload.get("answer_text", "")
    if not answer_text:
        return

    for start in range(0, len(answer_text), chunk_size):
        yield answer_text[start:start + chunk_size]