"""Logging callback handler attached via ``RunnableConfig``.

Emits structured log entries at key lifecycle points so that the existing
``json_logger`` pipeline can consume them alongside other events.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from app.logging.json_logger import get_logger

logger = get_logger("app.callbacks.logging_cb")


class LoggingCallbackHandler(BaseCallbackHandler):
    """A LangChain callback handler that logs structured events using the
    app's ``json_logger`` module.

    Attach via::

        config = RunnableConfig(callbacks=[LoggingCallbackHandler()])
        await chain.ainvoke(inputs, config)
    """

    def __init__(self, session_id: Optional[str] = None):
        super().__init__()
        self.session_id = session_id

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
        logger.info(
            "lcel_chain_start",
            extra={
                "run_id": str(run_id) if run_id else None,
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "session_id": self.session_id,
                "tags": tags,
                "metadata": metadata,
                "inputs_keys": list(inputs.keys()) if isinstance(inputs, dict) else None,
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
        logger.info(
            "lcel_chain_end",
            extra={
                "run_id": str(run_id) if run_id else None,
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "session_id": self.session_id,
                "tags": tags,
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
        logger.error(
            "lcel_chain_error",
            extra={
                "run_id": str(run_id) if run_id else None,
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "session_id": self.session_id,
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
        logger.info(
            "lcel_llm_start",
            extra={
                "run_id": str(run_id) if run_id else None,
                "session_id": self.session_id,
                "prompt_snippet": (prompts[0][:200] + "...") if prompts else None,
            },
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        logger.info(
            "lcel_llm_end",
            extra={
                "run_id": str(run_id) if run_id else None,
                "session_id": self.session_id,
                "token_usage": getattr(response, "token_usage", None),
                "model_name": getattr(response, "model_name", None),
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
        logger.error(
            "lcel_llm_error",
            extra={
                "run_id": str(run_id) if run_id else None,
                "session_id": self.session_id,
                "error": str(error),
                "error_type": type(error).__name__,
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
        logger.info(
            "lcel_tool_start",
            extra={
                "run_id": str(run_id) if run_id else None,
                "session_id": self.session_id,
                "tool_name": serialized.get("name") if isinstance(serialized, dict) else None,
                "input_snippet": input_str[:200] if input_str else None,
            },
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        logger.info(
            "lcel_tool_end",
            extra={
                "run_id": str(run_id) if run_id else None,
                "session_id": self.session_id,
                "output_snippet": output[:200] if output else None,
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
        logger.error(
            "lcel_tool_error",
            extra={
                "run_id": str(run_id) if run_id else None,
                "session_id": self.session_id,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )