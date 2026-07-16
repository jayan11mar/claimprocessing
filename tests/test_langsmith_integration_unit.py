import types

import app.langsmith_integration as lsi


def test_start_trace_uses_client_and_sets_trace_id(monkeypatch):
    class FakeRun:
        def __init__(self, id):
            self.id = id

    class FakeClient:
        def create_run(self, name=None, run_type=None, start_time=None, extra=None):
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
        def create_feedback(self, run_id, key, score=None, comment=None, source_info=None):
            calls["run_id"] = run_id
            calls["key"] = key

    lsi._client = FakeClient()
    lsi._enabled = True
    lsi._last_trace_id = "00000000-0000-0000-0000-000000000001"

    lsi.record_span("myspan", {"k": "v"})
    assert "run_id" in calls
    assert "myspan" in calls["key"]

    lsi._client = None
    lsi._enabled = False
    lsi._last_trace_id = None