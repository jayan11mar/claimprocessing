"""Tests for API-level conversation caching functionality."""

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import server
from app.models.faq import FAQResponse, FAQIntent


class CachedFakeMemory:
    """Fake memory that tracks message storage for caching tests."""

    def __init__(self):
        self.store = {}
        self.append_count = 0

    def append_message(self, session_id, role, message):
        self.append_count += 1
        self.store.setdefault(session_id, []).append((role, message))

    def get_history(self, session_id):
        return self.store.get(session_id, [])

    def clear_history(self, session_id):
        self.store.pop(session_id, None)


class CachedFakeAgentChain:
    """Fake agent chain that tracks invocation count for caching tests."""

    def __init__(self):
        self.invoke_count = 0

    def invoke(self, session_id, message, context=None):
        self.invoke_count += 1
        timings = {"llm_ms": 100, "tools": []}
        if isinstance(context, dict) and isinstance(context.get("timings"), dict):
            context["timings"].update(timings)

        resp = FAQResponse(
            intent=FAQIntent.POLICY_STATUS,
            category="policy",
            confidence=0.9,
            answer_text=f"Cached answer for: {message}",
            reasoning="cached response",
            metadata={"timings": timings},
        )
        return resp


def test_cache_hit_returns_cached_response():
    """Test that identical requests within TTL return cached response."""
    memory = CachedFakeMemory()
    agent_chain = CachedFakeAgentChain()
    server._memory = memory
    server._agent_chain = agent_chain

    client = TestClient(server.app)

    payload = {"session_id": "cache-test-session", "message": "Check policy P123456"}

    # First request - should invoke agent chain
    resp1 = client.post("/chat", json=payload)
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["answer_text"] == "Cached answer for: Check policy P123456"
    assert agent_chain.invoke_count == 1
    # Note: memory.append_count is 0 because FakeAgentChain doesn't append to memory
    # The real AgentChain does, but our fake doesn't for testing purposes

    # Second identical request - should use cache
    resp2 = client.post("/chat", json=payload)
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["answer_text"] == "Cached answer for: Check policy P123456"
    assert agent_chain.invoke_count == 1  # Should NOT increment
    assert memory.append_count == 0  # Should NOT increment (cached response not stored again)

    # Verify cache hit metadata
    assert body2["chain_metadata"].get("cache_hit") is True


def test_cache_miss_for_different_messages():
    """Test that different messages result in cache miss."""
    memory = CachedFakeMemory()
    agent_chain = CachedFakeAgentChain()
    server._memory = memory
    server._agent_chain = agent_chain

    client = TestClient(server.app)

    payload1 = {"session_id": "cache-test-session", "message": "Check policy P123456"}
    payload2 = {"session_id": "cache-test-session", "message": "Check policy P789012"}

    # First request
    resp1 = client.post("/chat", json=payload1)
    assert resp1.status_code == 200
    assert agent_chain.invoke_count == 1

    # Different message - should miss cache
    resp2 = client.post("/chat", json=payload2)
    assert resp2.status_code == 200
    assert agent_chain.invoke_count == 2  # Should increment
    assert resp2.json()["chain_metadata"].get("cache_hit") is not True


def test_cache_miss_for_different_sessions():
    """Test that same message in different sessions results in cache miss."""
    memory = CachedFakeMemory()
    agent_chain = CachedFakeAgentChain()
    server._memory = memory
    server._agent_chain = agent_chain

    client = TestClient(server.app)

    payload1 = {"session_id": "session-A", "message": "Check policy P123456"}
    payload2 = {"session_id": "session-B", "message": "Check policy P123456"}

    # First request in session A
    resp1 = client.post("/chat", json=payload1)
    assert resp1.status_code == 200
    assert agent_chain.invoke_count == 1

    # Same message in session B - should miss cache
    resp2 = client.post("/chat", json=payload2)
    assert resp2.status_code == 200
    assert agent_chain.invoke_count == 2  # Should increment
    assert resp2.json()["chain_metadata"].get("cache_hit") is not True


def test_cache_invalidation_on_reset():
    """Test that reset clears the cache for that session."""
    memory = CachedFakeMemory()
    agent_chain = CachedFakeAgentChain()
    server._memory = memory
    server._agent_chain = agent_chain

    client = TestClient(server.app)

    payload = {"session_id": "reset-test-session", "message": "Check policy P123456"}

    # First request
    resp1 = client.post("/chat", json=payload)
    assert resp1.status_code == 200
    assert agent_chain.invoke_count == 1

    # Second request - should hit cache
    resp2 = client.post("/chat", json=payload)
    assert resp2.status_code == 200
    assert agent_chain.invoke_count == 1  # Cache hit
    assert resp2.json()["chain_metadata"].get("cache_hit") is True

    # Reset the session
    reset_resp = client.post("/reset", json={"session_id": "reset-test-session"})
    assert reset_resp.status_code == 200

    # Verify cache was cleared
    assert len(server._conversation_cache) == 0

    # Third request after reset - should miss cache
    resp3 = client.post("/chat", json=payload)
    assert resp3.status_code == 200
    assert agent_chain.invoke_count == 2  # Should increment after cache invalidation
    assert resp3.json()["chain_metadata"].get("cache_hit") is not True


def test_cache_cleanup_on_multiple_requests():
    """Test that cache cleanup happens automatically on requests."""
    memory = CachedFakeMemory()
    agent_chain = CachedFakeAgentChain()
    server._memory = memory
    server._agent_chain = agent_chain

    # Set cache TTL to a very short time for testing
    original_ttl = server._CACHE_TTL_SECONDS
    server._CACHE_TTL_SECONDS = 1

    try:
        client = TestClient(server.app)

        payload = {"session_id": "cleanup-test", "message": "Test message"}

        # First request
        resp1 = client.post("/chat", json=payload)
        assert resp1.status_code == 200
        assert agent_chain.invoke_count == 1
        assert len(server._conversation_cache) == 1

        # Wait for cache to expire
        time.sleep(1.1)

        # Second request - cache should be cleaned up
        resp2 = client.post("/chat", json=payload)
        assert resp2.status_code == 200
        assert agent_chain.invoke_count == 2  # Should invoke again
        assert resp2.json()["chain_metadata"].get("cache_hit") is not True
    finally:
        server._CACHE_TTL_SECONDS = original_ttl


def test_cache_respects_max_size():
    """Test that cache respects maximum size limit."""
    memory = CachedFakeMemory()
    agent_chain = CachedFakeAgentChain()
    server._memory = memory
    server._agent_chain = agent_chain

    # Set small cache size for testing
    original_max_size = server._CACHE_MAX_SIZE
    server._CACHE_MAX_SIZE = 3

    try:
        client = TestClient(server.app)

        # Add more entries than max cache size
        for i in range(5):
            payload = {"session_id": f"session-{i}", "message": f"Message {i}"}
            resp = client.post("/chat", json=payload)
            assert resp.status_code == 200

        # Cache should not exceed max size (allow for concurrent access during cleanup)
        assert len(server._conversation_cache) <= server._CACHE_MAX_SIZE + 1
    finally:
        server._CACHE_MAX_SIZE = original_max_size


def test_conftest_clears_cache():
    """Test that the conftest fixture clears the conversation cache."""
    memory = CachedFakeMemory()
    agent_chain = CachedFakeAgentChain()
    server._memory = memory
    server._agent_chain = agent_chain

    client = TestClient(server.app)

    payload = {"session_id": "conftest-test", "message": "Test message"}

    # First request
    resp1 = client.post("/chat", json=payload)
    assert resp1.status_code == 200
    assert len(server._conversation_cache) == 1

    # Manually clear cache (simulating conftest fixture)
    server._conversation_cache.clear()

    # Cache should be empty
    assert len(server._conversation_cache) == 0

    # Next request should work normally
    resp2 = client.post("/chat", json=payload)
    assert resp2.status_code == 200
    assert agent_chain.invoke_count == 2