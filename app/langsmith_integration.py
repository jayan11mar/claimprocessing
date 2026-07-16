"""LangSmith tracing integration for the claims processing application.

This module provides a thin wrapper around the LangSmith Client that is
compatible with langsmith SDK v0.4.x. The SDK's public API changed significantly
in the 0.4.x line — methods like ``start_run``, ``end_run``, ``log_event``,
``record_span`` do NOT exist on the client in this version.

We adapt by using:
  - ``create_run`` / ``update_run`` for trace lifecycle
  - ``create_feedback`` as a lightweight span-recording mechanism (only when
    a real LangSmith run UUID is available)
"""

from contextlib import contextmanager
import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
import uuid

# Load .env file at module import time
load_dotenv()

logger = logging.getLogger(__name__)

_enabled = False
_client = None
_last_trace_id = None

try:
    from langsmith import Client as LangSmithClient
    _LS_AVAILABLE = True
except ImportError:
    LangSmithClient = None
    _LS_AVAILABLE = False


def init_langsmith():
    global _enabled, _client
    api_key = os.environ.get("LANGSMITH_API_KEY")
    tracing_flag = os.environ.get("LANGSMITH_TRACING")
    project = os.environ.get("LANGSMITH_PROJECT_NAME")
    if (
        not api_key
        or not _LS_AVAILABLE
        or not tracing_flag
        or tracing_flag.lower() not in ("1", "true", "yes")
    ):
        _enabled = False
        return None

    def _create_client(*args, **kwargs):
        try:
            return LangSmithClient(*args, **kwargs)
        except (TypeError, ValueError) as exc:
            logger.warning("langsmith_client_init_failed", {"error": str(exc)})
            return None

    _client = _create_client(api_key=api_key)
    if _client is None:
        _client = _create_client(api_key=api_key, project=project)
    if _client is None:
        _client = _create_client(api_key)

    if _client is None:
        _enabled = False
        return None

    _enabled = True
    logger.info("langsmith_initialized", {"project": project})
    return _client


def _is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    if not value:
        return False
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def get_langsmith_trace_id():
    return _last_trace_id


@contextmanager
def start_trace(name: str):
    """Context manager that creates a LangSmith run for the duration of the block.

    Compatible with langsmith SDK v0.4.x — uses ``create_run`` / ``update_run``
    instead of the removed ``start_run`` / ``end_run`` methods.

    Yields a dict with ``trace_id`` (a string) so callers can attach spans or
    metadata to the same trace.
    """
    global _last_trace_id
    if not _enabled or _client is None:
        yield {"trace_id": None}
        return

    run = None
    try:
        run = _client.create_run(
            name=name,
            run_type="chain",
            start_time=datetime.now(timezone.utc),
            extra={"trace_name": name},
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning("langsmith_create_run_failed", {"name": name, "error": str(exc)})

    run_id = None
    if run is not None:
        run_id = getattr(run, "id", None)
        if run_id is None:
            run_id = getattr(run, "run_id", None)

    # Convert UUID objects to string; fall back if we don't have a real UUID
    if run_id is not None:
        trace_id = str(run_id)
        _last_trace_id = trace_id
    else:
        # create_run failed (e.g. network issue, invalid API key)
        # We set a non-UUID fallback so callers can still see a trace_id,
        # but spans will be silently dropped rather than causing errors.
        fallback = f"ls-{name}"
        _last_trace_id = fallback
        yield {"trace_id": fallback}
        return

    yield {"trace_id": trace_id}

    # Mark the run as completed on context exit
    if _client is not None:
        try:
            _client.update_run(
                run_id=run_id,
                end_time=datetime.now(timezone.utc),
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("langsmith_update_run_failed", {"run_id": trace_id, "error": str(exc)})


def record_span(name: str, metadata: dict) -> None:
    """Record a span / sub-step in the current trace using LangSmith ``create_feedback``.

    In langsmith v0.4.x there is no direct ``record_span`` or ``add_span`` API.
    We use ``create_feedback`` as a side-channel to attach span-like metadata
    to the active trace, keyed by ``_last_trace_id``.

    If tracing is disabled, no trace is active, or the trace ID is not a valid
    UUID (e.g. fallback mode when create_run failed), this is a silent no-op.
    """
    if not _enabled or _client is None or _last_trace_id is None:
        return None

    # create_feedback requires a real UUID — silently skip non-UUID trace IDs
    # (e.g. the "ls-{name}" fallback when create_run failed)
    if not _is_valid_uuid(_last_trace_id):
        return None

    try:
        _client.create_feedback(
            run_id=_last_trace_id,
            key=f"span:{name}",
            score=None,
            comment=None,
            source_info={"metadata": metadata},
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning("langsmith_create_feedback_failed", {"span": name, "error": str(exc)})

    return None


init_langsmith()