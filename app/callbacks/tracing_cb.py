"""Tracing callback handler attached via ``RunnableConfig``.

Integrates with the existing ``langsmith_integration`` module so that LCEL
invocations produce uniform traces alongside the legacy ``start_trace`` /
``record_span`` calls.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from app.langsmith_integration import record_span

logger = logging.getLogger(__name__)


class TracingCallbackHandler(BaseCallbackHandler):
    """A LangChain callback handler that records spans through the existing
    ``app.langsmith_integration`` module.

    Attach via::

        config = RunnableConfig(callbacks=[TracingCallbackHandler()])
        await chain.ainvoke(inputs, config)
    """

    def __init__(self, session_id: Optional[str] = None, trace_id: Optional[str] = None):
        super().__init__()
        self.session_id = session_id
        self.trace_id = trace_id

    # ── Chain lifecycle ──────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        span_name = metadata.get("lcel_span_name", "lcel_chain") if metadata else "lcel_chain"
        if self.trace_id:
            record_span(
                span_name,
                {
                    "session_id": self.session_id,
                    "run_id": str(run_id) if run_id else None,
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "lcel_event": "chain_start",
                },
            )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        span_name = kwargs.pop("lcel_span_name", "lcel_chain") if kwargs else "lcel_chain"
        if self.trace_id:
            record_span(
                f"{span_name}:end",
                {
                    "session_id": self.session_id,
                    "run_id": str(run_id) if run_id else None,
                    "lcel_event": "chain_end",
                },
            )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if self.trace_id:
            record_span(
                "lcel_chain:error",
                {
                    "session_id": self.session_id,
                    "run_id": str(run_id) if run_id else None,
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
            )

    # ── LLM lifecycle ────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if self.trace_id:
            record_span(
                "lcel_llm",
                {
                    "session_id": self.session_id,
                    "run_id": str(run_id) if run_id else None,
                    "lcel_event": "llm_start",
                },
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if self.trace_id:
            record_span(
                "lcel_llm:error",
                {
                    "session_id": self.session_id,
                    "run_id": str(run_id) if run_id else None,
                    "error": str(error),
                },
            )

    # ── Tool lifecycle ───────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if self.trace_id:
            record_span(
                "lcel_tool",
                {
                    "session_id": self.session_id,
                    "run_id": str(run_id) if run_id else None,
                    "tool_name": serialized.get("name") if isinstance(serialized, dict) else None,
                    "lcel_event": "tool_start",
                },
            )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if self.trace_id:
            record_span(
                "lcel_tool:error",
                {
                    "session_id": self.session_id,
                    "run_id": str(run_id) if run_id else None,
                    "error": str(error),
                },
            )