from typing import Any, Dict, Optional

from app.config import get_settings
from app.rag.embeddings import get_embedding_fn
from app.rag.qa_chain import run_qa_chain


def knowledge_retrieval(
    query: str,
    top_k: int = 3,
    claim_context: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the knowledge-base retrieval QA flow and return answer text plus citations.

    Args:
        query: The query string.
        top_k: Number of results to return.
        claim_context: Optional claim context string for answer formatting.
        metadata_filter: Optional dict of metadata fields to filter chunks on.
                         For example: {"doc_type": "policy_wording", "insurance_type": "health"}

    Returns:
        Dict with answer_text, citations, confidence, and retrieval_trace.
    """
    settings = get_settings()
    embedding_model = settings.EMBEDDING_MODEL or settings.OPENAI_EMBEDDING_MODEL
    embedding_fn = get_embedding_fn(embedding_model)
    payload = run_qa_chain(query, claim_context=claim_context, top_k=top_k, embedding_fn=embedding_fn, metadata_filter=metadata_filter)
    return {
        "answer_text": payload.get("answer_text", ""),
        "citations": payload.get("citations", []),
        "confidence": payload.get("confidence", 0.0),
        "retrieval_trace": [
            {
                "tool": "knowledge_retrieval",
                "query": query,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
                "result_count": len(payload.get("citations", [])),
            }
        ],
    }
