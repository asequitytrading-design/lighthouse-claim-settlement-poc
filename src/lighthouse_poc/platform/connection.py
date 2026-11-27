"""SQLAlchemy engine for a Databricks SQL warehouse, authenticated by databricks-sdk.

databricks-sdk unified auth covers every place the platform tooling runs: a notebook or job
(runtime identity), a developer laptop (CLI profile, OAuth U2M) and CI (service principal,
OAuth M2M). No personal access token is stored or put in the connection URL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..exceptions import ConfigurationError

HTTP_PATH = "/sql/1.0/warehouses/{warehouse_id}"


@dataclass(frozen=True)
class WarehouseTarget:
    host: str              # hostname without scheme
    warehouse_id: str
    catalog: str

    @property
    def http_path(self) -> str:
        return HTTP_PATH.format(warehouse_id=self.warehouse_id)

    def url(self) -> str:
        return f"databricks://{self.host}?http_path={self.http_path}&catalog={self.catalog}"


def resolve_warehouse(workspace_client: Any, warehouse_id: str | None) -> str:
    """The configured warehouse, else a running one, else the first available."""
    if warehouse_id:
        return warehouse_id
    warehouses = list(workspace_client.warehouses.list())
    if not warehouses:
        raise ConfigurationError("No SQL warehouse in this workspace; create one or set "
                                 "DATABRICKS_WAREHOUSE_ID")
    running = [w for w in warehouses if str(getattr(w, "state", "")).endswith("RUNNING")]
    return (running or warehouses)[0].id


class WarehouseEngineFactory:
    """Builds the engine; the workspace client and engine builder are injectable for tests."""

    def __init__(self, logger: logging.Logger, workspace_client: Any = None,
                 create_engine: Callable[..., Any] | None = None):
        self._logger = logger
        self._client = workspace_client
        self._create_engine = create_engine

    def _workspace(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient
            self._client = WorkspaceClient()
        return self._client

    def target(self, catalog: str, warehouse_id: str | None = None) -> WarehouseTarget:
        client = self._workspace()
        host = client.config.host.replace("https://", "").replace("http://", "").rstrip("/")
        return WarehouseTarget(host=host, warehouse_id=resolve_warehouse(client, warehouse_id),
                               catalog=catalog)

    def engine(self, catalog: str, warehouse_id: str | None = None):
        target = self.target(catalog, warehouse_id)
        config = self._workspace().config
        create_engine = self._create_engine
        if create_engine is None:
            from sqlalchemy import create_engine
        self._logger.info("warehouse_engine_created", extra={
            "warehouse_id": target.warehouse_id, "catalog": catalog})
        # credentials_provider: a zero-argument callable returning a header factory;
        # Config.authenticate refreshes OAuth tokens as needed.
        return create_engine(target.url(),
                             connect_args={"credentials_provider": lambda: config.authenticate},
                             pool_pre_ping=True)
