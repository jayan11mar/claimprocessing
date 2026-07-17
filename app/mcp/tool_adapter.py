"""MCP Tool Adapter — wraps MCP server tools as LangChain StructuredTools."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from app.mcp.client import MCPClient, MCPClientPool, MCPInvocationError, MCPTimeoutError
from app.mcp.registry import ServerDefinition, ToolSchema, get_registry

logger = logging.getLogger("app.mcp.tool_adapter")


def _build_pydantic_model(tool_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Build a Pydantic model from a JSON Schema input definition."""
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    fields: Dict[str, tuple] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _json_type_to_python(prop_schema.get("type", "string"))
        description = prop_schema.get("description", "")
        if prop_name in required_fields:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            fields[prop_name] = (Optional[py_type], Field(None, description=description))

    return create_model(f"{tool_name}Args", **fields)


def _json_type_to_python(json_type: str) -> type:
    mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(json_type, str)


def create_mcp_tool(
    server_key: str,
    tool_schema: ToolSchema,
    server_def: ServerDefinition,
    client_pool: MCPClientPool,
) -> StructuredTool:
    """Create a LangChain StructuredTool from an MCP tool schema."""

    args_model = _build_pydantic_model(tool_schema.name, tool_schema.input_schema)

    async def _arun(**kwargs: Any) -> str:
        client = client_pool.get(server_key)
        if client is None:
            client = MCPClient(server_def)
            client_pool.register(server_key, client)

        try:
            result = await client.invoke_tool(tool_schema.name, kwargs)
            return json.dumps(result, default=str)
        except (MCPInvocationError, MCPTimeoutError) as exc:
            logger.error("mcp_tool_error", {"tool": tool_schema.name, "error": str(exc)})
            return json.dumps({"error": str(exc), "tool": tool_schema.name})

    def _run(**kwargs: Any) -> str:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context — create a new event loop in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _arun(**kwargs))
                return future.result(timeout=30)
        else:
            return asyncio.run(_arun(**kwargs))

    return StructuredTool(
        name=tool_schema.name,
        description=tool_schema.description,
        args_schema=args_model,
        func=_run,
        coroutine=_arun,
    )


def discover_and_create_tools(client_pool: MCPClientPool) -> List[StructuredTool]:
    """Discover all MCP tools from the registry and create LangChain StructuredTools."""
    registry = get_registry()
    tools: List[StructuredTool] = []

    for server_key, server_def in registry.list_servers().items():
        for tool_schema in server_def.tools:
            tool = create_mcp_tool(server_key, tool_schema, server_def, client_pool)
            tools.append(tool)

    return tools