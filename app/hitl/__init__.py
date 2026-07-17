"""Human-In-The-Loop (HITL) module.

Provides:
- ``app/hitl/models.py`` — Pydantic models for HITL tasks
- ``app/hitl/store.py`` — SQLite-backed persistent task store
- ``app/hitl/triggers.py`` — Trigger rule evaluation engine
- ``app/hitl/manager.py`` — Pause / resume lifecycle orchestration
"""