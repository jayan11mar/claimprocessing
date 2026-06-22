#!/usr/bin/env python3
import os
import sys
import uuid

os.environ.setdefault("OPENAI_API_KEY", "test")

from app.memory.sqlite_memory import SQLiteMemory


def main() -> int:
    mem = SQLiteMemory()
    session_id = f"test-{uuid.uuid4()}"
    turns = 10
    expected = []

    for i in range(1, turns + 1):
        user = f"User message {i}"
        mem.append_message(session_id, "user", user)
        expected.append(("user", user))

        assistant = f"Assistant reply {i}"
        mem.append_message(session_id, "assistant", assistant)
        expected.append(("assistant", assistant))

    mem2 = SQLiteMemory()
    history = mem2.get_history(session_id)

    if len(history) != len(expected):
        print(f"FAIL: expected {len(expected)} messages, got {len(history)}")
        sys.exit(1)

    for idx, (role, content) in enumerate(expected):
        msg = history[idx]
        if msg.content != content:
            print(
                f"FAIL: mismatch at index {idx}: expected '{content}', got '{msg.content}'"
            )
            sys.exit(1)

    print("PASS: SQLiteMemory persisted and retrieved multi-turn conversation correctly.")

    mem2.clear_history(session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
