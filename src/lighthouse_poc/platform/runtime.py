"""Resolve platform names and the warehouse engine from a profile (CLI, notebook, jobs)."""
from __future__ import annotations

import logging

from ..config import ModelRoute, Profile
from ..exceptions import ConfigurationError
from ..gateway import GatewayChatClient, GatewayEmbeddingClient, build_auth
from .connection import WarehouseEngineFactory
from .naming import PlatformNames


def platform_names(profile: Profile, catalog: str | None = None,
                   prefix: str | None = None) -> PlatformNames:
    if profile.platform is None and not catalog:
        raise ConfigurationError(f"Profile '{profile.name}' has no 'platform' section; the AI "
                                 "Marketplace data platform runs on the databricks profile")
    base = profile.platform
    return PlatformNames(catalog=catalog or base.catalog,
                         prefix=base.schema_prefix if prefix is None and base else (prefix or ""))


def warehouse_engine(profile: Profile, names: PlatformNames, logger: logging.Logger,
                     warehouse_id: str | None = None, workspace_client=None):
    configured = profile.platform.warehouse_id if profile.platform else None
    factory = WarehouseEngineFactory(logger, workspace_client=workspace_client)
    return factory.engine(names.catalog, warehouse_id or configured)


def catalog_route(entry, params: dict | None = None) -> ModelRoute:
    """A gateway route for a model catalog entry (logical key -> Unity Catalog model service)."""
    return ModelRoute(alias=entry.key, upstream_model=entry.model_service,
                      params=params or {}, task=entry.task)


def catalog_chat_client(profile: Profile, entry, logger: logging.Logger, max_tokens: int = 1024,
                        timeout_seconds: int = 120):
    # reasoning models (qwen3.5) spend tokens before answering: leave room for that
    route = catalog_route(entry, {"temperature": 0, "max_tokens": max_tokens})
    return GatewayChatClient(profile.base_url_for(route), route, profile.gateway.kind,
                             build_auth(profile.gateway.auth), logger,
                             timeout_seconds=timeout_seconds, max_retries=1)


def catalog_embedding_client(profile: Profile, entry, logger: logging.Logger,
                             timeout_seconds: int = 120):
    route = catalog_route(entry)
    return GatewayEmbeddingClient(profile.base_url_for(route), route, profile.gateway.kind,
                                  build_auth(profile.gateway.auth), logger,
                                  timeout_seconds=timeout_seconds, max_retries=1)
