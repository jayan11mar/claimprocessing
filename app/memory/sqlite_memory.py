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
