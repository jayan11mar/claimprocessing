from fastapi.testclient import TestClient

from app.api import server


class DummyMemory:
    def get_history(self, session_id):
        return []

    def clear_history(self, session_id):
        pass


def test_health_endpoint(monkeypatch):
    server._memory = DummyMemory()
    client = TestClient(server.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_reset_endpoint(monkeypatch):
    cleared = {}

    class Mem:
        def clear_history(self, session_id):
            cleared["id"] = session_id

    server._memory = Mem()
    client = TestClient(server.app)
    resp = client.post("/reset", json={"session_id": "s1"})
    assert resp.status_code == 200
    assert cleared.get("id") == "s1"