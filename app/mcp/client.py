"""MCP Client — HTTP client for invoking MCP server tools with health checks, retries, and timeouts."""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import httpx

from app.mcp.registry import RetryConfig, ServerDefinition
from app.mcp.auth import build_auth_headers, mask_sensitive_headers

logger = logging.getLogger("app.mcp.client")


class MCPClientError(Exception):
    """Base exception for MCP client errors."""


class MCPHealthCheckError(MCPClientError):
    """Raised when a health check fails."""


class MCPTimeoutError(MCPClientError):
    """Raised when a request times out."""


class MCPInvocationError(MCPClientError):
    """Raised when a tool invocation fails."""


def _compute_backoff(retry: RetryConfig, attempt: int) -> float:
    """Compute exponential backoff with jitter for a given attempt (0-indexed)."""
    import random
    delay = min(retry.base_delay * (2 ** attempt), retry.max_delay)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter


class MCPClient:
    """Client for a single MCP server with health-check, retry, and timeout support."""

    def __init__(self, server: ServerDefinition) -> None:
        self.server = server
        self._base_url = server.url.rstrip("/")
        self._timeout = server.timeout_seconds
        self._retry = server.retry
        self._auth_headers = build_auth_headers(server.auth)
        self._last_health_check: Optional[float] = None
        self._health_check_interval: float = 30.0

    async def health_check(self, force: bool = False) -> bool:
        """Check if the server is healthy. Caches result for _health_check_interval seconds."""
        now = time.time()
        if not force and self._last_health_check is not None:
            if now - self._last_health_check < self._health_check_interval:
                return True

        url = f"{self._base_url}{self.server.health_endpoint}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                resp = await client.get(url, timeout=self._timeout)
            if resp.status_code == 200:
                self._last_health_check = now
                return True
            logger.warning("mcp_health_check_failed",
                           {"server": self.server.key, "status": resp.status_code, "body": resp.text[:200]})
            return False
        except httpx.TimeoutException:
            logger.warning("mcp_health_check_timeout", {"server": self.server.key, "url": url})
            raise MCPHealthCheckError(f"Health check timeout for {self.server.key}")
        except httpx.RequestError as exc:
            logger.warning("mcp_health_check_error",
                           {"server": self.server.key, "error": str(exc)})
            raise MCPHealthCheckError(f"Health check error for {self.server.key}: {exc}")

    async def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool on the MCP server with retry-with-backoff."""
        url = f"{self._base_url}{self.server.invoke_endpoint}"
        payload = {"tool": tool_name, "arguments": arguments}
        headers = dict(self._auth_headers)
        headers["Content-Type"] = "application/json"

        last_exception: Optional[Exception] = None
        for attempt in range(self._retry.max_retries):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                    resp = await client.post(url, json=payload, headers=headers, timeout=self._timeout)
                if resp.status_code < 500:
                    if resp.status_code == 200:
                        return resp.json()
                    body = resp.text[:500]
                    raise MCPInvocationError(f"Tool '{tool_name}' returned {resp.status_code}: {body}")

                logger.warning("mcp_invoke_retry",
                               {"server": self.server.key, "tool": tool_name,
                                "attempt": attempt + 1, "status": resp.status_code})
                last_exception = MCPInvocationError(f"Tool '{tool_name}' returned {resp.status_code}")

            except httpx.TimeoutException:
                logger.warning("mcp_invoke_timeout",
                               {"server": self.server.key, "tool": tool_name, "attempt": attempt + 1})
                last_exception = MCPTimeoutError(f"Tool '{tool_name}' timed out after {self._timeout}s")
                if attempt < self._retry.max_retries - 1:
                    await asyncio.sleep(_compute_backoff(self._retry, attempt))
                continue
            except httpx.RequestError as exc:
                logger.warning("mcp_invoke_request_error",
                               {"server": self.server.key, "tool": tool_name,
                                "attempt": attempt + 1, "error": str(exc)})
                last_exception = MCPInvocationError(f"Tool '{tool_name}' request failed: {exc}")
                if attempt < self._retry.max_retries - 1:
                    await asyncio.sleep(_compute_backoff(self._retry, attempt))
                continue

            if attempt < self._retry.max_retries - 1:
                await asyncio.sleep(_compute_backoff(self._retry, attempt))

        raise last_exception or MCPInvocationError(f"Tool '{tool_name}' invocation failed after {self._retry.max_retries} retries")

    async def discover_tools(self) -> list[Dict[str, Any]]:
        """Discover tools from the MCP server's /tools endpoint."""
        url = f"{self._base_url}{self.server.tools_endpoint}"
        headers = dict(self._auth_headers)
        headers["Content-Type"] = "application/json"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
                resp = await client.get(url, headers=headers, timeout=self._timeout)
            if resp.status_code == 200:
                data: Dict[str, Any] = resp.json()
                return data.get("tools", [])
            logger.warning("mcp_discover_tools_failed",
                           {"server": self.server.key, "status": resp.status_code})
            return []
        except Exception as exc:
            logger.warning("mcp_discover_tools_error",
                           {"server": self.server.key, "error": str(exc)})
            return []

    async def close(self) -> None:
        pass


class SyncMCPClient:
    """Synchronous wrapper around MCPClient for use in sync contexts."""

    def __init__(self, server: ServerDefinition) -> None:
        self._client = MCPClient(server)

    def health_check(self, force: bool = False) -> bool:
        return asyncio.run(self._client.health_check(force=force))

    def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self._client.invoke_tool(tool_name, arguments))

    def discover_tools(self) -> list[Dict[str, Any]]:
        return asyncio.run(self._client.discover_tools())

    def close(self) -> None:
        asyncio.run(self._client.close())


class MCPClientPool:
    """Manages a pool of MCP clients, one per server."""

    def __init__(self) -> None:
        self._clients: Dict[str, MCPClient] = {}

    def register(self, key: str, client: MCPClient) -> None:
        self._clients[key] = client

    def get(self, key: str) -> Optional[MCPClient]:
        return self._clients.get(key)

    def list_clients(self) -> Dict[str, MCPClient]:
        return dict(self._clients)

    async def close_all(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()


_pool_instance: Optional[MCPClientPool] = None


def get_client_pool() -> MCPClientPool:
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = MCPClientPool()
    return _pool_instance


def reset_client_pool() -> None:
    global _pool_instance
    _pool_instance = None