"""Additional tests for app/memory/sqlite_memory.py to achieve >85% coverage.

Covers branches missed by existing test_sqlite_persistence.py:
- get_history_records
- clear_history
- trim_history
- get_message_count
- Module-level functions (get_history, append_message, clear_history, get_message_count)
- reset_memory_singleton
- Edge cases in append_message and get_history
"""

from unittest.mock import patch, MagicMock
import pytest

from app.memory.sqlite_memory import (
    SQLiteMemory,
    get_history,
    append_message,
    clear_history,
    get_message_count,
    reset_memory_singleton,
    _get_memory_instance,
)


class TestSQLiteMemoryGetHistoryRecords:
    def test_get_history_records_returns_list_of_dicts(self):
        memory = SQLiteMemory()
        session_id = "test-history-records"
        memory.clear_history(session_id)

        memory.append_message(session_id, "user", "Hello")
        memory.append_message(session_id, "assistant", "Hi there!")

        records = memory.get_history_records(session_id)
        assert len(records) == 2
        assert records[0] == {"role": "user", "content": "Hello"}
        assert records[1] == {"role": "assistant", "content": "Hi there!"}

    def test_get_history_records_empty_session(self):
        memory = SQLiteMemory()
        session_id = "test-empty-records"
        memory.clear_history(session_id)

        records = memory.get_history_records(session_id)
        assert records == []


class TestSQLiteMemoryClearHistory:
    def test_clear_history_removes_all_messages(self):
        memory = SQLiteMemory()
        session_id = "test-clear-history"
        memory.clear_history(session_id)

        memory.append_message(session_id, "user", "Hello")
        memory.append_message(session_id, "assistant", "Hi")
        assert len(memory.get_history(session_id)) == 2

        memory.clear_history(session_id)
        assert len(memory.get_history(session_id)) == 0

    def test_clear_history_nonexistent_session(self):
        memory = SQLiteMemory()
        # Should not raise
        memory.clear_history("nonexistent-session")


class TestSQLiteMemoryTrimHistory:
    def test_trim_history_removes_oldest_messages(self):
        memory = SQLiteMemory()
        session_id = "test-trim-history"
        memory.clear_history(session_id)

        # Add 6 messages (3 turns)
        for i in range(3):
            memory.append_message(session_id, "user", f"Message {i}")
            memory.append_message(session_id, "assistant", f"Response {i}")

        assert len(memory.get_history(session_id)) == 6

        # Trim to 2 turns (4 messages)
        memory.trim_history(session_id, max_turns=2)
        history = memory.get_history(session_id)
        assert len(history) == 4
        # The oldest messages should be removed
        assert "Message 0" not in [m.content for m in history]

    def test_trim_history_no_op_when_under_limit(self):
        memory = SQLiteMemory()
        session_id = "test-trim-no-op"
        memory.clear_history(session_id)

        memory.append_message(session_id, "user", "Hello")
        memory.append_message(session_id, "assistant", "Hi")

        memory.trim_history(session_id, max_turns=10)
        assert len(memory.get_history(session_id)) == 2

    def test_trim_history_empty_session(self):
        memory = SQLiteMemory()
        # Should not raise
        memory.trim_history("nonexistent-session", max_turns=5)


class TestSQLiteMemoryGetMessageCount:
    def test_get_message_count_returns_correct_count(self):
        memory = SQLiteMemory()
        session_id = "test-message-count"
        memory.clear_history(session_id)

        assert memory.get_message_count(session_id) == 0

        memory.append_message(session_id, "user", "Hello")
        assert memory.get_message_count(session_id) == 1

        memory.append_message(session_id, "assistant", "Hi")
        assert memory.get_message_count(session_id) == 2

    def test_get_message_count_nonexistent_session(self):
        memory = SQLiteMemory()
        count = memory.get_message_count("nonexistent-session")
        assert count == 0


class TestSQLiteMemoryGetHistoryEdgeCases:
    def test_get_history_returns_empty_list_for_nonexistent_session(self):
        memory = SQLiteMemory()
        session_id = "test-nonexistent-history"
        memory.clear_history(session_id)

        history = memory.get_history(session_id)
        assert history == []

    def test_get_history_returns_base_messages(self):
        memory = SQLiteMemory()
        session_id = "test-base-messages"
        memory.clear_history(session_id)

        memory.append_message(session_id, "user", "Hello")
        memory.append_message(session_id, "assistant", "Hi")
        memory.append_message(session_id, "system", "System msg")

        history = memory.get_history(session_id)
        assert len(history) == 3
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
        assert isinstance(history[2], SystemMessage)

    def test_get_history_unknown_role_defaults_to_human(self):
        memory = SQLiteMemory()
        session_id = "test-unknown-role"
        memory.clear_history(session_id)

        # Direct DB insert to test unknown role
        import sqlite3
        from datetime import datetime
        with sqlite3.connect(memory.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "bot", "Bot message", datetime.utcnow().isoformat()),
            )
            conn.commit()

        history = memory.get_history(session_id)
        assert len(history) == 1
        from langchain_core.messages import HumanMessage
        assert isinstance(history[0], HumanMessage)
        assert history[0].content == "Bot message"


class TestModuleLevelFunctions:
    def test_get_history_module_level(self):
        session_id = "test-module-get-history"
        # Clear first
        clear_history(session_id)
        append_message(session_id, "user", "Hello")
        append_message(session_id, "assistant", "Hi")

        history = get_history(session_id)
        assert len(history) == 2

    def test_append_message_module_level(self):
        session_id = "test-module-append"
        clear_history(session_id)
        append_message(session_id, "user", "Test message")

        count = get_message_count(session_id)
        assert count == 1

    def test_clear_history_module_level(self):
        session_id = "test-module-clear"
        append_message(session_id, "user", "Hello")
        clear_history(session_id)

        count = get_message_count(session_id)
        assert count == 0

    def test_get_message_count_module_level(self):
        session_id = "test-module-count"
        clear_history(session_id)
        assert get_message_count(session_id) == 0

        append_message(session_id, "user", "Hello")
        assert get_message_count(session_id) == 1

    def test_reset_memory_singleton(self):
        # Get the singleton instance
        instance1 = _get_memory_instance()
        reset_memory_singleton()
        instance2 = _get_memory_instance()
        # After reset, a new instance should be created
        assert instance1 is not instance2

    def test_append_message_trims_history(self):
        """Module-level append_message should trim history to 10 turns."""
        session_id = "test-module-trim"
        clear_history(session_id)

        # Add 12 turns (24 messages)
        for i in range(12):
            append_message(session_id, "user", f"Message {i}")
            append_message(session_id, "assistant", f"Response {i}")

        # Should be trimmed to 10 turns (20 messages)
        count = get_message_count(session_id)
        assert count == 20