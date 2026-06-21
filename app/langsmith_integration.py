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
except Exception:
    LangSmithClient = None
    _LS_AVAILABLE = False


def init_langsmith():
    global _enabled, _client
    api_key = os.environ.get("LANGSMITH_API_KEY")
    tracing_flag = os.environ.get("LANGSMITH_TRACING")
    project = os.environ.get("LANGSMITH_PROJECT_NAME")
    if not api_key or not _LS_AVAILABLE or not tracing_flag or tracing_flag.lower() not in ("1", "true", "yes"):
        _enabled = False
        return None
    try:
        try:
            _client = LangSmithClient(api_key=api_key)
        except TypeError:
            try:
                _client = LangSmithClient(api_key=api_key, project=project)
            except Exception:
                _client = LangSmithClient(api_key)
        _enabled = True
        logger.info("langsmith_initialized", {"project": project})
        return _client
    except Exception as exc:
        logger.warning("langsmith_init_failed", {"error": str(exc)})
        _enabled = False
        return None


def get_langsmith_trace_id():
    return _last_trace_id


@contextmanager
def start_trace(name: str):
    global _last_trace_id
    if not _enabled:
        yield {"trace_id": None}
        return
    try:
        run_id = None
        if hasattr(_client, "start_run"):
            try:
                run = _client.start_run(name=name)
                run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
            except Exception:
                run_id = None
        elif hasattr(_client, "create_run"):
            try:
                run = _client.create_run(name=name)
                run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
            except Exception:
                run_id = None
        trace_id = run_id or f"ls-{name}"
        _last_trace_id = trace_id
        yield {"trace_id": trace_id}
    finally:
        try:
            if hasattr(_client, "end_run"):
                _client.end_run()
            elif hasattr(_client, "stop_run"):
                _client.stop_run()
        except Exception:
            pass


def record_span(name: str, metadata: dict):
    if not _enabled or _client is None:
        return None
    try:
        if hasattr(_client, "log_event"):
            _client.log_event({"span": name, "meta": metadata})
            return None
        for fn in ("add_span", "record_span", "log_span"):
            if hasattr(_client, fn):
                try:
                    getattr(_client, fn)(name, metadata)
                    return None
                except Exception:
                    continue
        if hasattr(_client, "log"):
            try:
                _client.log({"span": name, "meta": metadata})
            except Exception:
                pass
    except Exception:
        pass


init_langsmith()
