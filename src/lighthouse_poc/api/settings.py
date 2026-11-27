"""Settings of the demo API (environment variables; Databricks Apps sets them in app.yaml).

  LIGHTHOUSE_PROFILE        config profile (default databricks)
  LIGHTHOUSE_APP_USE_CASE   request tag lh_use_case of the app's model calls (default claims_copilot)
  LIGHTHOUSE_SESSION_STORE  sqlite | postgres | lakebase (default sqlite); the semantic cache
                            follows it (pgvector for postgres/lakebase, in-memory for sqlite)
  LIGHTHOUSE_SQLITE_PATH    sqlite file (default .lighthouse/sessions.db; on Apps the disk is ephemeral)
  LIGHTHOUSE_PG_DSN         postgres DSN for local Postgres or Lakebase (phase 5)
  LIGHTHOUSE_WEB_DIST       built UI to serve at / (default apps/lighthouse_demo/web/dist)
  LIGHTHOUSE_DASHBOARD_URL  optional link to the Lakeview observability dashboard
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import PROJECT_ROOT


@dataclass(frozen=True)
class AppSettings:
    profile: str = "databricks"
    use_case: str = "claims_copilot"
    session_store: str = "sqlite"
    sqlite_path: str = ".lighthouse/sessions.db"
    pg_dsn: str | None = None
    web_dist: Path = PROJECT_ROOT / "apps" / "lighthouse_demo" / "web" / "dist"
    dashboard_url: str | None = None

    @classmethod
    def from_env(cls, env: dict | None = None) -> "AppSettings":
        env = os.environ if env is None else env
        return cls(
            profile=env.get("LIGHTHOUSE_PROFILE", "databricks"),
            use_case=env.get("LIGHTHOUSE_APP_USE_CASE", "claims_copilot"),
            session_store=env.get("LIGHTHOUSE_SESSION_STORE", "sqlite").lower(),
            sqlite_path=env.get("LIGHTHOUSE_SQLITE_PATH", ".lighthouse/sessions.db"),
            pg_dsn=env.get("LIGHTHOUSE_PG_DSN") or None,
            web_dist=Path(env.get("LIGHTHOUSE_WEB_DIST") or cls.web_dist),
            dashboard_url=env.get("LIGHTHOUSE_DASHBOARD_URL") or None)
