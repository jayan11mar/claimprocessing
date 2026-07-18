"""Agent registry — typed descriptors loaded from config/agents.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from app.config import settings


@dataclass(frozen=True)
class AgentDescriptor:
    """Immutable descriptor for a child agent."""

    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    prompt_key: str = ""
    retrieval_scope: str = ""
    capabilities: List[str] = field(default_factory=list)


# ── Module-level cache ───────────────────────────────────────────────────────

_agents: Optional[dict[str, AgentDescriptor]] = None


def _load_agents() -> dict[str, AgentDescriptor]:
    """Load agent descriptors from the YAML config path."""
    path = settings.AGENTS_CONFIG_PATH
    if not os.path.isfile(path):
        return {}

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not data or "agents" not in data:
        return {}

    registry: dict[str, AgentDescriptor] = {}
    for entry in data["agents"]:
        desc = AgentDescriptor(
            name=entry["name"],
            description=entry.get("description", ""),
            tools=entry.get("tools", []),
            prompt_key=entry.get("prompt_key", ""),
            retrieval_scope=entry.get("retrieval_scope", ""),
            capabilities=entry.get("capabilities", []),
        )
        registry[desc.name] = desc

    return registry


def get_agent(name: str) -> AgentDescriptor | None:
    """Return the descriptor for *name*, or *None* if not found."""
    global _agents
    if _agents is None:
        _agents = _load_agents()
    return _agents.get(name)


def list_agents() -> list[AgentDescriptor]:
    """Return all registered agent descriptors."""
    global _agents
    if _agents is None:
        _agents = _load_agents()
    return list(_agents.values())