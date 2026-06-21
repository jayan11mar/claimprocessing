import types

import app.langsmith_integration as lsi


def test_start_trace_uses_client_and_sets_trace_id(monkeypatch):
    class FakeRun:
        def __init__(self, id):
            self.id = id

    class FakeClient:
        def start_run(self, name=None):
            return FakeRun(id=f"run-{name}")

    lsi._client = FakeClient()
    lsi._enabled = True

    with lsi.start_trace("test-span") as trace:
        assert trace["trace_id"] == "run-test-span"

    lsi._client = None
    lsi._enabled = False
    lsi._last_trace_id = None


def test_record_span_calls_log_event(monkeypatch):
    calls = {}

    class FakeClient:
        def log_event(self, payload):
            calls["payload"] = payload

    lsi._client = FakeClient()
    lsi._enabled = True

    lsi.record_span("myspan", {"k": "v"})
    assert "payload" in calls
    assert calls["payload"]["span"] == "myspan"
    assert calls["payload"]["meta"]["k"] == "v"

    lsi._client = None
    lsi._enabled = False
