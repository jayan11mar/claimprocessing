import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings


class SQLiteMemory:

    def __init__(self) -> None:
        settings = get_settings()
        self.db_path = Path(settings.SQLITE_DB_PATH)
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policies (
                    policy_number TEXT PRIMARY KEY,
                    policy_holder_id TEXT,
                    status TEXT NOT NULL,
                    sum_insured REAL NOT NULL,
                    deductible REAL NOT NULL,
                    copay_percent REAL NOT NULL,
                    sub_limits TEXT,
                    depreciation_schedule TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    product_code TEXT,
                    coverage_type TEXT,
                    underwriting_class TEXT,
                    risk_category TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    policy_number TEXT NOT NULL,
                    policy_holder_id TEXT,
                    claim_amount REAL NOT NULL,
                    incident_date TEXT,
                    admission_date TEXT,
                    discharge_date TEXT,
                    diagnosis_code TEXT,
                    hospital_name TEXT,
                    supporting_documents TEXT,
                    extra_info TEXT,
                    status TEXT,
                    loss_type TEXT,
                    reported_date TEXT,
                    closed_date TEXT,
                    approved_amount REAL,
                    fraud_score REAL,
                    settlement_status TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(policy_number) REFERENCES policies(policy_number)
                )
                """
            )
            conn.commit()

    def append_message(self, session_id: str, role: str, content: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_history(self, session_id: str) -> List[BaseMessage]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            rows = cursor.fetchall()

        messages: List[BaseMessage] = []
        for role, content in rows:
            role_lower = role.lower()
            if role_lower == "system":
                messages.append(SystemMessage(content=content))
            elif role_lower == "assistant":
                messages.append(AIMessage(content=content))
            elif role_lower == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        return messages

    def get_history_records(self, session_id: str) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            rows = cursor.fetchall()

        return [{"role": role, "content": content} for role, content in rows]

    def clear_history(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def trim_history(self, session_id: str, max_turns: int = 10) -> None:
        """Trim conversation history to keep only the most recent `max_turns` user+assistant pairs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE session_id = ?",
                (session_id,),
            )
            count = cursor.fetchone()[0]
            # Each turn = 2 messages (user + assistant)
            keep = max_turns * 2
            if count > keep:
                conn.execute(
                    """DELETE FROM chat_history
                       WHERE session_id = ? AND id IN (
                           SELECT id FROM chat_history
                           WHERE session_id = ?
                           ORDER BY id ASC
                           LIMIT ?
                       )""",
                    (session_id, session_id, count - keep),
                )
                conn.commit()

    def get_message_count(self, session_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE session_id = ?",
                (session_id,),
            )
            return cursor.fetchone()[0]


# ---------------------------------------------------------------------------
# Standalone module-level functions (LangChain-friendly interface)
# ---------------------------------------------------------------------------

def get_history(session_id: str) -> List[BaseMessage]:
    """Retrieve conversation history for a session as a list of BaseMessage objects."""
    memory = _get_memory_instance()
    return memory.get_history(session_id)


def append_message(session_id: str, role: str, content: str) -> None:
    """Append a single message to the conversation history for a session."""
    memory = _get_memory_instance()
    memory.append_message(session_id, role, content)
    memory.trim_history(session_id, max_turns=10)


def clear_history(session_id: str) -> None:
    """Clear all history for a session."""
    memory = _get_memory_instance()
    memory.clear_history(session_id)


def get_message_count(session_id: str) -> int:
    """Return the number of stored messages for a session."""
    memory = _get_memory_instance()
    return memory.get_message_count(session_id)


def _get_memory_instance() -> SQLiteMemory:
    """Return a singleton-like SQLiteMemory instance for module-level access."""
    if not hasattr(_get_memory_instance, "_instance"):
        _get_memory_instance._instance = SQLiteMemory()
    return _get_memory_instance._instance


def reset_memory_singleton() -> None:
    """Clear the cached singleton so next call creates a fresh instance."""
    if hasattr(_get_memory_instance, "_instance"):
        del _get_memory_instance._instance
