from fastapi.testclient import TestClient

from app.api import server
from app.models.faq import FAQResponse, FAQIntent


class FakeMemory:
    def __init__(self):
        self.store = {}

    def append_message(self, session_id, role, message):
        self.store.setdefault(session_id, []).append((role, message))

    def get_history(self, session_id):
        return self.store.get(session_id, [])

    def clear_history(self, session_id):
        self.store.pop(session_id, None)


class FakeAgentChain:
    def invoke(self, session_id, message, context=None):
        timings = {"llm_ms": 100, "tools": [{"tool": "claims_intake", "ms": 50}]}
        if isinstance(context, dict) and isinstance(context.get("timings"), dict):
            context["timings"].update(timings)

        resp = FAQResponse(
            intent=FAQIntent.POLICY_STATUS,
            category="claims",
            confidence=0.9,
            answer_text=f"Mocked answer for {message}",
            reasoning="mocked",
            metadata={"timings": timings},
        )
        return resp


def test_chat_chain_metadata_contains_timings(monkeypatch):
    server._memory = FakeMemory()
    server._agent_chain = FakeAgentChain()

    client = TestClient(server.app)

    payload = {"session_id": "test-session", "message": "What is the status of claim P101?"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert "chain_metadata" in body
    chain_meta = body["chain_metadata"]
    assert isinstance(chain_meta.get("latency_ms"), int)
    assert chain_meta.get("llm_ms") == 100
    assert isinstance(chain_meta.get("tool_timings"), list)
    assert chain_meta["tool_timings"][0]["tool"] == "claims_intake"
    assert chain_meta["latency_target_ms"] == 8000
    assert chain_meta["latency_within_target"] is True

    structured = body.get("structured", {})
    assert structured.get("metadata", {}).get("timings", {})["llm_ms"] == 100


def test_chat_latency_target_for_simple_query(monkeypatch):
    class SimpleAgentChain(FakeAgentChain):
        def invoke(self, session_id, message, context=None):
            timings = {"llm_ms": 50, "tools": []}
            if isinstance(context, dict) and isinstance(context.get("timings"), dict):
                context["timings"].update(timings)
            return FAQResponse(
                intent=FAQIntent.POLICY_STATUS,
                category="claims",
                confidence=0.9,
                answer_text="Simple answer",
                reasoning="mocked",
                metadata={"timings": timings},
            )

    server._memory = FakeMemory()
    server._agent_chain = SimpleAgentChain()

    client = TestClient(server.app)
    payload = {"session_id": "simple-session", "message": "What is the policy status?"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    chain_meta = body["chain_metadata"]
    assert chain_meta["latency_target_ms"] == 3000
    assert chain_meta["latency_within_target"] is True
    assert chain_meta["is_tool_augmented"] is False


def test_chat_latency_target_for_tool_augmented_query(monkeypatch):
    server._memory = FakeMemory()
    server._agent_chain = FakeAgentChain()

    client = TestClient(server.app)
    payload = {"session_id": "tool-session", "message": "Register a claim"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    chain_meta = body["chain_metadata"]
    assert chain_meta["latency_target_ms"] == 8000
    assert chain_meta["latency_within_target"] is True
    assert chain_meta["is_tool_augmented"] is True


def test_chat_latency_target_for_simple_query(monkeypatch):
    class SimpleAgentChain(FakeAgentChain):
        def invoke(self, session_id, message, context=None):
            timings = {"llm_ms": 50, "tools": []}
            if isinstance(context, dict) and isinstance(context.get("timings"), dict):
                context["timings"].update(timings)
            return FAQResponse(
                intent=FAQIntent.POLICY_STATUS,
                category="claims",
                confidence=0.9,
                answer_text="Simple answer",
                reasoning="mocked",
                metadata={"timings": timings},
            )

    server._memory = FakeMemory()
    server._agent_chain = SimpleAgentChain()

    client = TestClient(server.app)
    payload = {"session_id": "simple-session", "message": "What is the policy status?"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    chain_meta = body["chain_metadata"]
    assert chain_meta["latency_target_ms"] == 3000
    assert chain_meta["latency_within_target"] is True
    assert chain_meta["is_tool_augmented"] is False
