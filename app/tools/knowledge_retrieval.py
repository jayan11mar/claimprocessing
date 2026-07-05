from typing import Any, Dict, Optional

from app.rag.qa_chain import run_qa_chain


def knowledge_retrieval(query: str, top_k: int = 3, claim_context: Optional[str] = None) -> Dict[str, Any]:
    """Run the knowledge-base retrieval QA flow and return answer text plus citations."""
    payload = run_qa_chain(query, claim_context=claim_context, top_k=top_k)
    return {
        "answer_text": payload.get("answer_text", ""),
        "citations": payload.get("citations", []),
        "confidence": payload.get("confidence", 0.0),
        "retrieval_trace": [
            {
                "tool": "knowledge_retrieval",
                "query": query,
                "top_k": top_k,
                "result_count": len(payload.get("citations", [])),
            }
        ],
    }
