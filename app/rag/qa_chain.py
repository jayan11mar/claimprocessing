"""A lightweight QA chain that produces LLM-synthesized answer text with [chunk_id] citations from hybrid retrieval."""

import logging
from typing import Any, Dict, List, Optional

from app.rag.chunkers import Chunk, ChunkConfig, chunk_document
from app.rag.loaders import load_documents_from_manifest
from app.rag.retriever_hybrid import hybrid_retrieve
from app.rag.vectorstores import get_vector_store
from app.rag.vectorstores.base import VectorStore
from app.prompt_manager.registry import get_registry

logger = logging.getLogger(__name__)


def _get_rag_system_prompt(
    claim_context: Optional[str],
    query: str,
    context_str: str,
) -> str:
    """Get the RAG QA system prompt from the versioned registry.

    Falls back to the old inline version if the registry is unavailable.
    """
    try:
        registry = get_registry()
        if claim_context:
            return registry.get_template("rag_qa")
        return registry.get_template("rag_qa")
    except Exception:
        return (
            "You are a helpful insurance claims assistant. Answer the user's question based solely on "
            "the provided context chunks. "
            "Every factual statement MUST be followed by a citation in the format [chunk_id] referencing "
            "the chunk that supports it. "
            "If the context does not contain enough information to answer the question, say so clearly. "
            "Do not make up information. Do not use external knowledge."
        )


# ── Module-level persisted chunks & vector store cache ───────────────────────
_PERSISTED_CHUNKS: Optional[List[Chunk]] = None
_PERSISTED_VECTOR_STORE: Optional[VectorStore] = None


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


def _get_persisted_vector_store() -> Optional[VectorStore]:
    """
    Return the persisted FAISS vector store, or None if unavailable.

    Cached at module level so it is loaded only once.
    """
    global _PERSISTED_VECTOR_STORE
    if _PERSISTED_VECTOR_STORE is not None:
        return _PERSISTED_VECTOR_STORE

    from app.config import get_settings
    from app.rag.vectorstores.faiss_store import FAISSStore

    persist_path = get_settings().VECTOR_PERSIST_PATH
    store = FAISSStore.load(persist_path)
    if store is not None and store.chunk_count > 0:
        _PERSISTED_VECTOR_STORE = store
    return _PERSISTED_VECTOR_STORE


# ── Module-level LLM cache so we reuse the same ChatOpenAI instance ──────────
import time as _time
_LLM_CACHE: Optional[Any] = None
_LLM_CACHE_TIME: float = 0.0
_LLM_CACHE_TTL: float = 3600.0  # Re-create hourly


def _get_cached_llm():
    """Return a cached ChatOpenAI instance, creating one if needed."""
    global _LLM_CACHE, _LLM_CACHE_TIME
    now = _time.time()
    if _LLM_CACHE is None or (now - _LLM_CACHE_TIME) > _LLM_CACHE_TTL:
        from app.config import get_settings
        from langchain_openai import ChatOpenAI
        settings = get_settings()
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your-"):
            _LLM_CACHE = None
        else:
            _LLM_CACHE = ChatOpenAI(
                model=settings.OPENAI_MODEL_NAME,
                temperature=0.3,
                openai_api_key=settings.OPENAI_API_KEY,
                timeout=settings.OPENAI_REQUEST_TIMEOUT,
                max_retries=2,
            )
        _LLM_CACHE_TIME = now
    return _LLM_CACHE


def _llm_synthesize_answer(
    query: str,
    results: List[Dict[str, Any]],
    claim_context: Optional[str] = None,
) -> str:
    """Send reranked chunks to the app LLM and get a synthesized answer with [chunk_id] citations.

    Args:
        query: The user query string.
        results: Reranked list of result dicts (must have ``chunk_id`` and ``chunk`` keys).
        claim_context: Optional claim context string.

    Returns:
        Synthesized answer text with inline [chunk_id] citations, or a fallback
        excerpt if the LLM is unavailable.
    """
    from app.config import get_settings

    settings = get_settings()

    # ── Fallback when no LLM is configured ───────────────────────────────
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your-"):
        best_chunk = results[0]["chunk"]
        excerpt = best_chunk.text.strip().replace("\n", " ")
        if len(excerpt) > 260:
            excerpt = excerpt[:257] + "..."
        if claim_context:
            return f"For {claim_context}, the retrieved guidance says: {excerpt}"
        return f"The retrieved guidance says: {excerpt}"

    # ── Build context from reranked chunks ───────────────────────────────
    context_parts: List[str] = []
    for result in results:
        chunk_id = result["chunk_id"]
        text = result["chunk"].text.strip()
        context_parts.append(f"[{chunk_id}] {text}")
    context_str = "\n\n".join(context_parts)

    system_prompt = _get_rag_system_prompt(claim_context, query, context_str)
    if claim_context:
        user_prompt = f"Claim context: {claim_context}\n\nQuestion: {query}\n\nContext:\n{context_str}"
    else:
        user_prompt = f"Question: {query}\n\nContext:\n{context_str}"

    llm = _get_cached_llm()
    if llm is None:
        best_chunk = results[0]["chunk"]
        excerpt = best_chunk.text.strip().replace("\n", " ")
        if len(excerpt) > 260:
            excerpt = excerpt[:257] + "..."
        if claim_context:
            return f"For {claim_context}, the retrieved guidance says: {excerpt}"
        return f"The retrieved guidance says: {excerpt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.warning("LLM synthesis failed, falling back to excerpt: %s", exc)
        best_chunk = results[0]["chunk"]
        excerpt = best_chunk.text.strip().replace("\n", " ")
        if len(excerpt) > 260:
            excerpt = excerpt[:257] + "..."
        if claim_context:
            return f"For {claim_context}, the retrieved guidance says: {excerpt}"
        return f"The retrieved guidance says: {excerpt}"


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

    # Load the persistent vector store for dense retrieval (avoids re-embedding chunks)
    vector_store = _get_persisted_vector_store()

    results = hybrid_retrieve(
        chunks, query, k=top_k,
        embedding_fn=embedding_fn,
        metadata_filter=metadata_filter,
        vector_store=vector_store,
    )
    if not results:
        return {
            "answer_text": "No relevant guidance was found in the knowledge base.",
            "citations": [],
            "confidence": 0.0,
        }

    # ── Build citations list from reranked results ───────────────────────
    citations = []
    for result in results[: min(3, len(results))]:
        rerank_score = result.get("rerank_score")
        if rerank_score is None:
            logger.warning(
                "Missing rerank_score on chunk %s; combined_score=%s used instead.",
                result.get("chunk_id"),
                result.get("combined_score"),
            )
            rerank_score = result["combined_score"]
        citations.append(
            {
                "chunk_id": result["chunk_id"],
                "text": result["chunk"].text,
                "source_id": result["source_id"],
                "source_path": result["source_path"],
                "rerank_score": rerank_score,
                "score": rerank_score,
            }
        )

    # ── LLM-synthesize answer with [chunk_id] citations ──────────────────
    answer_text = _llm_synthesize_answer(query, results, claim_context=claim_context)

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
    """Run a hybrid retrieval QA flow and return LLM-synthesized answer text plus citations.

    Every factual claim in the answer references a [chunk_id] for traceability.

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