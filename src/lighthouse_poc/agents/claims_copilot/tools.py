"""The claims copilot's tool set, wired from the profile (Databricks platform section).

Names come from the platform config (catalog + schema prefix): claims data and the tool function
in <prefix>ai_claims, the knowledge index in <prefix>ai_knowledge, the Genie space from
platform.genie_space_id or found by title (create it with `lighthouse platform genie ensure`).
"""
from __future__ import annotations

import logging
from typing import Any

from ...config import Profile, require
from ...exceptions import ConfigurationError
from ..core.tools import Tool
from ..tools.databricks_tools import ClaimCaseTool, GuidanceSearchTool, PortfolioTool
from ..tools.mcp_client import McpHttpClient, genie_path, uc_functions_path, vector_search_path


def build_tools(profile: Profile, logger: logging.Logger, auth: Any = None, embedder: Any = None,
                workspace: Any = None, http: Any = None) -> dict[str, Tool]:
    from ...gateway import build_auth
    from ...platform.genie import GenieSpaces, space_title
    from ...platform.runtime import platform_names
    platform = profile.platform
    if platform is None:
        raise ConfigurationError(f"Profile '{profile.name}' has no platform section (Databricks only)")
    names = platform_names(profile)
    host = require(profile.gateway.base_url, "gateway.base_url").split("/ai-gateway")[0]
    auth = auth or build_auth(profile.gateway.auth)
    if embedder is None:
        from ...gateway import build_embedding_model
        embedder = build_embedding_model(profile, platform.embedding_model, logger)
    space_id = platform.genie_space_id
    if not space_id:
        if workspace is None:
            from databricks.sdk import WorkspaceClient
            workspace = WorkspaceClient()
        space_id = GenieSpaces(workspace, logger).find(space_title(names))
    tools: dict[str, Tool] = {
        "get_claim_case": ClaimCaseTool(McpHttpClient(
            host, uc_functions_path(names.catalog, names.claims), auth, logger, http=http)),
        "search_guidance": GuidanceSearchTool(McpHttpClient(
            host, vector_search_path(names.catalog, names.knowledge), auth, logger, http=http), embedder),
    }
    if space_id:
        tools["ask_portfolio"] = PortfolioTool(McpHttpClient(
            host, genie_path(space_id), auth, logger, timeout_seconds=90, http=http))
    else:
        logger.warning("portfolio_tool_unavailable", extra={
            "reason": f"no Genie space '{space_title(names)}': run lighthouse platform genie ensure"})
    return tools
