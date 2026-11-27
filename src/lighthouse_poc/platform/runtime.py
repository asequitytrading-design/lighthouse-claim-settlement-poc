"""Resolve platform names and the warehouse engine from a profile (CLI, notebook, jobs)."""
from __future__ import annotations

import logging

from ..config import Profile
from ..exceptions import ConfigurationError
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
