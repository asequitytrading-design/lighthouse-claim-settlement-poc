"""Live settings of a marketplace app, stored in <mkt>.app_config (revision 0006).

One definition of the settings, used by the writer (notebook 06) and the reader (the app):
  session_store    sqlite | lakebase            where chats and the semantic cache live
  default_model    an app model of the catalog  the copilot's default (the composer can override)
  semantic_cache   on | off                     guidance answers from the cache
  cache_threshold  0.85 .. 0.99                 cosine similarity for a cache hit
  grounding_check  on | off                     answerability + quote check on guidance
  genie            on | off                     the ask_portfolio tool (Genie)
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

from ..exceptions import ConfigurationError
from .naming import PlatformNames
from .sql import literal


@dataclass(frozen=True)
class AppConfig:
    session_store: str = "sqlite"
    default_model: str = "claims-alt-qwen-next"
    semantic_cache: str = "on"
    cache_threshold: str = "0.95"
    grounding_check: str = "on"
    genie: str = "on"

    def as_dict(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


ON_OFF = {"on", "off"}
STORES = {"sqlite", "lakebase"}


def validate(key: str, value: str, models: set[str] | None = None) -> str:
    value = str(value).strip().lower() if key != "default_model" else str(value).strip()
    if key == "session_store" and value in STORES:
        return value
    if key in ("semantic_cache", "grounding_check", "genie") and value in ON_OFF:
        return value
    if key == "cache_threshold":
        try:
            number = float(value)
        except ValueError:
            number = -1
        if 0.85 <= number <= 0.99:
            return f"{number:.2f}"
    if key == "default_model" and value and (models is None or value in models):
        return value
    allowed = {"session_store": STORES, "semantic_cache": ON_OFF, "grounding_check": ON_OFF,
               "genie": ON_OFF, "cache_threshold": "0.85-0.99",
               "default_model": sorted(models) if models else "an app model key"}
    if key not in allowed:
        raise ConfigurationError(f"Unknown app setting {key!r}")
    raise ConfigurationError(f"Invalid value {value!r} for {key}: allowed {allowed[key]}")


def merge_rows(base: AppConfig, rows: list[dict], models: set[str] | None = None) -> AppConfig:
    """Apply stored rows over the defaults; invalid rows are ignored (reported by the caller)."""
    values = {}
    for row in rows:
        try:
            values[row["config_key"]] = validate(row["config_key"], row["config_value"], models)
        except ConfigurationError:
            continue
    return replace(base, **{k: v for k, v in values.items() if k in base.as_dict()})


def read_sql(names: PlatformNames, app_name: str) -> str:
    return (f"SELECT config_key, config_value, updated_by, updated_at FROM {names.table('app_config')} "
            f"WHERE app_name = {literal(app_name)}")


def upsert_sql(names: PlatformNames, app_name: str, settings: dict[str, str], user: str,
               models: set[str] | None = None) -> str:
    """MERGE of validated settings (raises ConfigurationError before writing anything)."""
    clean = {key: validate(key, value, models) for key, value in settings.items()}
    rows = ",\n  ".join(f"({literal(app_name)}, {literal(k)}, {literal(v)}, {literal(user)})"
                        for k, v in clean.items())
    return (f"MERGE INTO {names.table('app_config')} t USING (SELECT * FROM VALUES\n  {rows}\n"
            f"  AS v(app_name, config_key, config_value, updated_by)) s\n"
            f"ON t.app_name = s.app_name AND t.config_key = s.config_key\n"
            f"WHEN MATCHED THEN UPDATE SET t.config_value = s.config_value, t.updated_by = s.updated_by, "
            f"t.updated_at = current_timestamp()\n"
            f"WHEN NOT MATCHED THEN INSERT (app_name, config_key, config_value, updated_by, updated_at) "
            f"VALUES (s.app_name, s.config_key, s.config_value, s.updated_by, current_timestamp())")
