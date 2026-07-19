"""Metrics callback handler attached via ``RunnableConfig``.

Collects latency and call-count metrics per lifecycle phase and exposes a
``report()`` method for the API to include in ``chain_metadata``.
"""

import time
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler


class MetricsCallbackHandler(BaseCallbackHandler):
    """A LangChain callback handler that accumulates latency and call-count
    metrics per node type (chain, LLM, tool).

    Metrics are stored in a public ``self.metrics`` dict that can be read at
    any time, e.g. from the endpoint handler::

        config = RunnableConfig(callbacks=[MetricsCallbackHandler()])
        result = chain.invoke(inputs, config)
        report = handler.report()  # {"chain": {"count": 1, "total_ms": ...}, ...}

    Notes
    -----
        - Times are recorded as wall-clock milliseconds.
        - ``report()`` returns a snapshot; the internal counter continues
          accumulating if further invocations share the same handler.
    """

    def __init__(self):
        super().__init__()
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._start_times: Dict[str, float] = {}

    def _ensure(self, category: str) -> None:
        if category not in self.metrics:
            self.metrics[category] = {"count": 0, "total_ms": 0.0, "errors": 0}

    def _record_start(self, run_id: Any, category: str) -> None:
        self._start_times[f"{category}:{run_id}"] = time.time()

    def _record_end(self, run_id: Any, category: str) -> None:
        key = f"{category}:{run_id}"
        start = self._start_times.pop(key, None)
        if start is not None:
            elapsed_ms = (time.time() - start) * 1000.0
            self._ensure(category)
            self.metrics[category]["count"] += 1
            self.metrics[category]["total_ms"] += elapsed_ms

    # ── Chain lifecycle ──────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._record_start(run_id, "chain")

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._record_end(run_id, "chain")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._record_end(run_id, "chain")
            self._ensure("chain")
            self.metrics["chain"]["errors"] += 1

    # ── LLM lifecycle ────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._record_start(run_id, "llm")

    def on_llm_end(self, *, run_id: Any = None, **kwargs: Any) -> None:
        if run_id is not None:
            self._record_end(run_id, "llm")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._record_end(run_id, "llm")
            self._ensure("llm")
            self.metrics["llm"]["errors"] += 1

    # ── Tool lifecycle ───────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._record_start(run_id, "tool")

    def on_tool_end(self, *, run_id: Any = None, **kwargs: Any) -> None:
        if run_id is not None:
            self._record_end(run_id, "tool")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._record_end(run_id, "tool")
            self._ensure("tool")
            self.metrics["tool"]["errors"] += 1

    # ── Public accessors ─────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Return a snapshot of the current metrics.

        Returns:
            A dict like ``{"chain": {"count": 1, "total_ms": 12.34, "errors": 0}, ...}``
        """
        return dict(self.metrics)