"""MCP Client Pool - singleton shared by all Agents.

Avoids resource leaks from creating MultiServerMCPClient on every Agent.__call__.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.billing_agent import UserIdInjector

logger = logging.getLogger(__name__)

_pool: "MCPClientPool | None" = None


class MCPClientPool:
    """Global MCP client pool shared across all Agents."""

    def __init__(self, servers_config: dict[str, Any], agent_dir: str) -> None:
        self._servers_config = servers_config
        self._agent_dir = agent_dir
        self._client: MultiServerMCPClient | None = None
        self._all_tools: list | None = None

    async def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        for _name, cfg in self._servers_config.get("mcpServers", {}).items():
            if cfg.get("cwd") and cfg["cwd"] == ".":
                cfg["cwd"] = self._agent_dir
            if cfg.get("command") and cfg["command"] == "python":
                cfg["command"] = sys.executable
        self._client = MultiServerMCPClient(
            connections=self._servers_config.get("mcpServers", {}),
            tool_interceptors=[UserIdInjector()],
        )
        self._all_tools = await self._client.get_tools()
        logger.info("MCPClientPool: connected, %d tools", len(self._all_tools))

    # Agent -> allowed tool whitelist (RBAC for MCP tools)
    AGENT_TOOL_WHITELIST = {
        "product_agent": {"get_promotable_products", "search_product_catalog", "get_promotion_materials"},
        "billing_agent": {"query_user_orders", "query_user_instances"},
        "promotion_agent": {"get_promotable_products", "search_product_catalog", "get_promotion_materials", "generate_ai_poster"},
        "recommendation_agent": {"get_promotable_products", "search_product_catalog", "get_promotion_materials"},
        "finops_agent": {"query_user_instances", "analyze_instance_usage"},
    }

    async def get_tools(self, tool_names: list[str], caller_agent: str = "") -> list:
        await self._ensure_connected()
        assert self._all_tools is not None
        name_set = set(tool_names)

        if caller_agent and caller_agent in self.AGENT_TOOL_WHITELIST:
            allowed = self.AGENT_TOOL_WHITELIST[caller_agent]
            unauthorized = name_set - allowed
            if unauthorized:
                logger.warning(
                    "MCPPool RBAC: agent %s attempted unauthorized: %s",
                    caller_agent, unauthorized,
                )
            name_set = name_set & allowed

        tools = [t for t in self._all_tools if t.name in name_set]
        logger.debug("MCPPool: returning %d tools for agent %s", len(tools), caller_agent or "unknown")
        return tools

    async def close_all(self) -> None:
        self._client = None
        self._all_tools = None
        logger.info("MCPClientPool: closed")


async def get_mcp_pool() -> MCPClientPool:
    global _pool
    if _pool is not None:
        return _pool
    agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(agent_dir, "config", "mcp_servers.json")
    with open(config_path, "r", encoding="utf-8") as f:
        servers_config = json.load(f)
    _pool = MCPClientPool(servers_config, agent_dir)
    await _pool._ensure_connected()
    return _pool


async def close_mcp_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close_all()
        _pool = None