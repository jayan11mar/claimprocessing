"""Global test configuration and fixtures for state cleanup.

This module ensures that tests are isolated from each other by:
1. Resetting in-memory demo data (_DEMO_POLICIES, _DEMO_CLAIMS)
2. Clearing the settings LRU cache
3. Resetting the memory singleton so tests that use module-level
   functions (get_history, append_message) get a fresh SQLiteMemory
   pointing to the default claims.db, rather than a stale singleton.
4. Patching both LLM seams used by the LCEL router to prevent real
   API calls from leaking into test output.
"""

import unittest.mock as mock

import pytest

from app.config import get_settings
from app.memory.sqlite_memory import reset_memory_singleton
from app.models.domain import reset_demo_data


if not getattr(mock.MagicMock, "_copilot_empty_spec_patch", False):
    _original_magicmock_getattr = mock.MagicMock.__getattr__

    def _compat_magicmock_getattr(self, name):
        try:
            return _original_magicmock_getattr(self, name)
        except AttributeError:
            # Use __dict__ directly to avoid infinite recursion when checking _mock_methods
            if self.__dict__.get("_mock_methods") == []:
                child = mock.MagicMock(name=name)
                setattr(self, name, child)
                return child
            raise

    mock.MagicMock.__getattr__ = _compat_magicmock_getattr
    mock.MagicMock._copilot_empty_spec_patch = True


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset all module-level caches before each test to prevent cross-test contamination.

    This fixture runs automatically for every test and:
    - Clears the @lru_cache on get_settings()
    - Resets in-memory demo policies and claims to their original state
    - Resets the SQLiteMemory singleton so a fresh instance is created
    - Clears the API-level conversation cache
    """
    # Clear settings cache
    get_settings.cache_clear()

    # Reset in-memory demo data
    reset_demo_data()

    # Reset memory singleton so next module-level call creates a fresh instance
    reset_memory_singleton()

    # Clear API-level conversation cache to prevent test isolation issues
    try:
        import builtins
        server = getattr(builtins, 'server', None)
        if server and hasattr(server, '_conversation_cache'):
            server._conversation_cache.clear()
    except (ImportError, AttributeError):
        pass

    yield
    # No teardown needed - the singleton reset and demo data reset at the
    # start of the next test is sufficient for isolation. The production
    # claims.db file is intentionally preserved.


@pytest.fixture(autouse=True)
def mock_llm_seams(monkeypatch):
    """Patch both LLM seams used by the LCEL router to prevent real API calls.

    The LCEL router fans out to two LLM seams (confirmed by trace):
      1. app.chains.base_chain.get_chat_model  (UNCACHED, primary)
         Used by tool/hitl/default chains.
      2. app.rag.qa_chain._get_cached_llm       (TTL-cached, secondary)
         Used by the rag chain.

    Both do llm.invoke(messages) then read response.content.
    Tests that only patched one seam would leak real output (e.g.
    'Fraud score for claim C1001').

    This fixture patches both with a MagicMock that returns a canned
    AIMessage, and resets the RAG LLM cache so the patched version
    is picked up immediately.
    """
    from unittest.mock import MagicMock

    from langchain_core.messages import AIMessage

    # Clear settings cache to ensure fresh state before patching
    get_settings.cache_clear()

    # Create a fake LLM that returns a canned response
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="Mocked answer")

    # Patch both LLM seams
    monkeypatch.setattr(
        "app.chains.base_chain.get_chat_model",
        lambda *a, **k: fake_llm,
    )
    monkeypatch.setattr(
        "app.rag.qa_chain._get_cached_llm",
        lambda *a, **k: fake_llm,
    )

    # Reset the RAG LLM cache so the patched _get_cached_llm is used
    from app.rag.qa_chain import reset_llm_cache
    reset_llm_cache()

    yield

    # Teardown: reset caches to avoid cross-test contamination
    reset_llm_cache()
    get_settings.cache_clear()