from contextlib import contextmanager
import os
import logging
from dotenv import load_dotenv

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


def get_langsmith_trace_id():
    return _last_trace_id


@contextmanager
def start_trace(name: str):
    global _last_trace_id
    if not _enabled:
        yield {"trace_id": None}
        return

    def _safe_client_call(method_name: str, *args, **kwargs):
        if not hasattr(_client, method_name):
            return None
        method = getattr(_client, method_name)
        try:
            return method(*args, **kwargs)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("langsmith_method_failed", {"method": method_name, "error": str(exc)})
            return None

    run_id = None
    run = _safe_client_call("start_run", name=name)
    if run is None:
        run = _safe_client_call("create_run", name=name)

    if run is not None:
        run_id = getattr(run, "id", None) or getattr(run, "run_id", None)

    trace_id = run_id or f"ls-{name}"
    _last_trace_id = trace_id
    yield {"trace_id": trace_id}

    for method_name in ("end_run", "stop_run"):
        if getattr(_client, method_name, None) is None:
            continue
        try:
            getattr(_client, method_name)()
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("langsmith_shutdown_failed", {"method": method_name, "error": str(exc)})


def record_span(name: str, metadata: dict):
    if not _enabled or _client is None:
        return None

    try:
        log_event = getattr(_client, "log_event")
    except AttributeError:
        log_event = None

    if log_event is not None:
        try:
            log_event({"span": name, "meta": metadata})
            return None
        except AttributeError:
            pass
        except (TypeError, ValueError, RuntimeError) as exc:
            logger.warning("langsmith_log_event_failed", {"error": str(exc)})
            return None

    for fn in ("add_span", "record_span", "log_span"):
        if not hasattr(_client, fn):
            continue
        method = getattr(_client, fn)
        try:
            method(name, metadata)
            return None
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("langsmith_span_method_failed", {"method": fn, "error": str(exc)})

    if hasattr(_client, "log"):
        try:
            _client.log({"span": name, "meta": metadata})
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("langsmith_log_failed", {"error": str(exc)})
    return None


init_langsmith()
