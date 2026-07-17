"""MCP Auth — handles per-server authentication for MCP tool invocations."""

import base64
from typing import Dict, Optional

from app.mcp.registry import AuthConfig


def build_auth_headers(auth: AuthConfig) -> Dict[str, str]:
    """Build HTTP authentication headers for the given auth config."""
    headers: Dict[str, str] = {}

    if auth.type == "none" or not auth.type:
        return headers

    if auth.type == "api_key":
        key = auth.credentials.get("default", "")
        header_name = auth.header_name or "X-API-Key"
        headers[header_name] = key

    elif auth.type == "bearer":
        token = auth.credentials.get("token", "")
        headers["Authorization"] = f"Bearer {token}"

    elif auth.type == "basic":
        username = auth.credentials.get("username", "")
        password = auth.credentials.get("password", "")
        raw = f"{username}:{password}"
        encoded = base64.b64encode(raw.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"

    return headers


def mask_sensitive_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Mask sensitive header values for logging."""
    sensitive_keys = {"authorization", "x-api-key", "cookie"}
    masked = {}
    for k, v in headers.items():
        if k.lower() in sensitive_keys:
            masked[k] = v[:8] + "..." if len(v) > 8 else "***"
        else:
            masked[k] = v
    return masked