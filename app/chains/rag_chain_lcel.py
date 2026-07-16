"""LCEL chain that wraps the existing RAG QA logic (``run_qa_chain`` /
``knowledge_retrieval``) as a ``Runnable``.

This chain does **not** rewrite or replace ``app/rag/qa_chain.py`` or
``app/tools/knowledge_retrieval.py``. It simply wraps them in an LCEL
interface so they can be composed in the router.
"""

import logging
from typing import Any, Dict, Optional

from langchain_core.runnables import Runnable, RunnableLambda

from app.tools.knowledge_retrieval import knowledge_retrieval

logger = logging.getLogger(__name__)


def _run_rag(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the existing knowledge retrieval tool and return a result dict.

    Expected input keys:
        - ``user_message`` (str): the user query.
        - ``metadata`` (dict, optional): may contain ``metadata_filter``,
          ``claim_context``, ``top_k``.

    Returns:
        A dict with ``answer_text``, ``citations``, ``confidence``, and
        ``retrieval_trace``.
    """
    user_message: str = inputs.get("user_message", "")
    meta: Dict[str, Any] = inputs.get("metadata", {}) or {}

    metadata_filter: Optional[Dict[str, Any]] = meta.get("metadata_filter")
    claim_context: Optional[str] = meta.get("claim_context")
    top_k: int = int(meta.get("top_k", 3))

    result = knowledge_retrieval(
        query=user_message,
        top_k=top_k,
        claim_context=claim_context,
        metadata_filter=metadata_filter,
    )

    return {
        "answer_text": result.get("answer_text", ""),
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", 0.0),
        "retrieval_trace": result.get("retrieval_trace", []),
    }


# ── Public Runnable ─────────────────────────────────────────────────────────

rag_lcel_chain: Runnable = RunnableLambda(_run_rag)
"""A ``Runnable`` that wraps the existing RAG knowledge-retrieval pipeline.

Usage::

    result = rag_lcel_chain.invoke({
        "user_message": "What does the policy cover?",
        "metadata": {"metadata_filter": {"insurance_type": "health"}},
    })
    print(result["answer_text"])
"""