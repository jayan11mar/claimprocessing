"""MCP integration — Model Context Protocol for external tool discovery and invocation.

Modules:
    registry   — Reads config/mcp_servers.yaml and manages server definitions.
    auth       — Per-server authentication (none, api_key, bearer, basic).
    client     — HTTP client with health checks, retry-with-backoff, and timeouts.
    tool_adapter — Wraps MCP tools as LangChain StructuredTools.
    servers    — Stub MCP servers for offline reproducible testing.
"""