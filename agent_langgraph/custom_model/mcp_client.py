# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load MCP server configuration and expose LangChain-ready tools."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

DEFAULT_CONFIG_FILENAME = "mcp_servers.json"


def _ensure_path(config_path: str | None) -> Path:
    """Resolve the MCP config path, preferring env override when provided."""

    resolved_path = Path(
        config_path
        or os.environ.get("MCP_SERVERS_CONFIG_PATH")
        or Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"MCP server config not found at {resolved_path}. "
            "Provide MCP_SERVERS_CONFIG_PATH or create mcp_servers.json next to mcp_client.py "
            "(use mcp_servers.json.example as a starting point)."
        )

    return resolved_path


def _normalize_env(env: dict[str, Any] | None) -> dict[str, str] | None:
    """Coerce environment values to strings for subprocess invocation."""

    if env is None:
        return None
    return {key: str(value) for key, value in env.items()}


def _server_to_connection(server_name: str, server: dict[str, Any]) -> Connection:
    """Convert MCP server JSON to langchain-mcp-adapters connection config."""

    transport = server.get("transport") or server.get("type") or "stdio"

    if transport == "stdio":
        if not server.get("command"):
            msg = f"Server '{server_name}' is missing a command for stdio transport"
            raise ValueError(msg)

        return {
            "transport": "stdio",
            "command": server["command"],
            "args": server.get("args", []),
            "env": _normalize_env(server.get("env")),
            "timeout": server.get("timeout"),
        }

    if transport == "sse":
        return {
            "transport": "sse",
            "url": server["url"],
            "headers": server.get("headers"),
            "timeout": server.get("timeout"),
            "sse_read_timeout": server.get("sse_read_timeout"),
        }

    if transport == "websocket":
        return {
            "transport": "websocket",
            "url": server["url"],
        }

    if transport == "streamable_http":
        return {
            "transport": "streamable_http",
            "url": server["url"],
            "timeout": server.get("timeout"),
            "sse_read_timeout": server.get("sse_read_timeout"),
        }

    msg = f"Unsupported MCP transport '{transport}' in server '{server_name}'"
    raise ValueError(msg)


def load_mcp_connections(config_path: str | None = None) -> dict[str, Connection]:
    """Load MCP servers from JSON config into langchain-mcp-adapters connections."""

    config_file = _ensure_path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    servers: dict[str, Any] = config.get("mcpServers", {})
    connections: dict[str, Connection] = {}

    for name, server in servers.items():
        if server.get("disabled"):
            continue
        connections[name] = _server_to_connection(name, server)

    if not connections:
        raise ValueError(
            "No enabled MCP servers were found. Ensure at least one entry is enabled in the config."
        )

    return connections


def build_mcp_client(config_path: str | None = None) -> MultiServerMCPClient:
    """Create a MultiServerMCPClient from a JSON configuration file."""

    connections = load_mcp_connections(config_path)
    return MultiServerMCPClient(connections)


def load_mcp_tools(
    config_path: str | None = None, *, client: MultiServerMCPClient | None = None
) -> list[BaseTool]:
    """Load LangChain-compatible tools from configured MCP servers synchronously."""

    client = client or build_mcp_client(config_path)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(client.get_tools())
    finally:
        asyncio.set_event_loop(None)
        loop.close()
