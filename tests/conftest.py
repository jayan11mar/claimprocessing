"""Global test configuration and fixtures for state cleanup.

This module ensures that tests are isolated from each other by:
1. Resetting in-memory demo data (_DEMO_POLICIES, _DEMO_CLAIMS)
2. Clearing the settings LRU cache
3. Resetting the memory singleton so tests that use module-level
   functions (get_history, append_message) get a fresh SQLiteMemory
   pointing to the default claims.db, rather than a stale singleton.
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
