"""MCP Server Registry — reads config/mcp_servers.yaml and manages server definitions."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from app.config import get_settings


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 0.2
    max_delay: float = 2.0


@dataclass
class AuthConfig:
    type: str = "none"  # none, api_key, bearer, basic
    header_name: Optional[str] = None
    credentials: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ServerDefinition:
    key: str
    name: str
    description: str
    transport: str
    url: str
    health_endpoint: str
    tools_endpoint: str
    invoke_endpoint: str
    timeout_seconds: float
    retry: RetryConfig
    auth: AuthConfig
    tools: List[ToolSchema]


class MCPServerRegistry:
    """Reads and caches MCP server definitions from config/mcp_servers.yaml."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = get_settings().MCP_SERVERS_PATH
        self._config_path = config_path
        self._servers: Dict[str, ServerDefinition] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not os.path.exists(self._config_path):
            raise FileNotFoundError(f"MCP servers config not found: {self._config_path}")
        with open(self._config_path, "r") as f:
            raw = yaml.safe_load(f)
        servers_raw = raw.get("servers", {})
        for key, srv in servers_raw.items():
            retry_raw = srv.get("retry", {})
            auth_raw = srv.get("auth", {})
            tools_raw = srv.get("tools", [])
            tools = [
                ToolSchema(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("input_schema", {"type": "object", "properties": {}}),
                )
                for t in tools_raw
            ]
            self._servers[key] = ServerDefinition(
                key=key,
                name=srv.get("name", key),
                description=srv.get("description", ""),
                transport=srv.get("transport", "http"),
                url=srv.get("url", ""),
                health_endpoint=srv.get("health_endpoint", "/health"),
                tools_endpoint=srv.get("tools_endpoint", "/tools"),
                invoke_endpoint=srv.get("invoke_endpoint", "/invoke"),
                timeout_seconds=float(srv.get("timeout_seconds", 2.0)),
                retry=RetryConfig(
                    max_retries=int(retry_raw.get("max_retries", 3)),
                    base_delay=float(retry_raw.get("base_delay", 0.2)),
                    max_delay=float(retry_raw.get("max_delay", 2.0)),
                ),
                auth=AuthConfig(
                    type=auth_raw.get("type", "none"),
                    header_name=auth_raw.get("header_name"),
                    credentials=auth_raw.get("credentials", {}),
                ),
                tools=tools,
            )
        self._loaded = True

    def list_servers(self) -> Dict[str, ServerDefinition]:
        self._ensure_loaded()
        return dict(self._servers)

    def get_server(self, key: str) -> Optional[ServerDefinition]:
        self._ensure_loaded()
        return self._servers.get(key)

    def get_all_tools(self) -> List[tuple[str, ToolSchema]]:
        """Return list of (server_key, tool_schema) for all tools across all servers."""
        self._ensure_loaded()
        result: List[tuple[str, ToolSchema]] = []
        for key, srv in self._servers.items():
            for tool in srv.tools:
                result.append((key, tool))
        return result

    def find_tool(self, tool_name: str) -> Optional[tuple[str, ToolSchema, ServerDefinition]]:
        """Find a tool by name across all servers."""
        self._ensure_loaded()
        for key, srv in self._servers.items():
            for tool in srv.tools:
                if tool.name == tool_name:
                    return (key, tool, srv)
        return None


_registry_instance: Optional[MCPServerRegistry] = None


def get_registry() -> MCPServerRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = MCPServerRegistry()
    return _registry_instance


def reset_registry() -> None:
    global _registry_instance
    _registry_instance = None