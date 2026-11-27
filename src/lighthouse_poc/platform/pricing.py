"""The rate card: config/model_pricing.yaml -> <mkt>.model_price (effective-dated).

Prices are reference data like the model list, so they are synced, not migrated. Each rate has an
effective_from date; the view v_model_price derives effective_to from the next rate of the same
model, and the cost views price every call with the rate in force on the call's date. The table is
a mirror of the YAML (one MERGE: insert new, update changed, delete removed), because the YAML
itself holds the history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..config import PROJECT_ROOT
from ..exceptions import ConfigurationError
from .model_catalog import ModelEntry
from .naming import PlatformNames
from .sql import literal

PRICING_FILE = PROJECT_ROOT / "config" / "model_pricing.yaml"
PRICE_SOURCES = {"illustrative", "list", "contract"}


@dataclass(frozen=True)
class ModelPrice:
    model_key: str
    effective_from: str              # YYYY-MM-DD
    input_per_1m_tokens: float
    output_per_1m_tokens: float
    price_source: str
    currency: str = "USD"
    notes: str | None = None


def _number(raw: dict, name: str, where: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigurationError(f"{where}: '{name}' must be a number >= 0")
    return float(value)


def _price(raw: dict, index: int, currency: str, known: set[str] | None) -> ModelPrice:
    where = f"rates[{index}] ({raw.get('model_key', '?')})"
    key = raw.get("model_key")
    if not key:
        raise ConfigurationError(f"{where}: 'model_key' is required")
    if known is not None and key not in known:
        raise ConfigurationError(f"{where}: model_key is not in config/model_catalog.yaml")
    try:
        effective = date.fromisoformat(str(raw.get("effective_from"))).isoformat()
    except ValueError:
        raise ConfigurationError(f"{where}: effective_from must be YYYY-MM-DD") from None
    source = raw.get("price_source")
    if source not in PRICE_SOURCES:
        raise ConfigurationError(f"{where}: price_source must be one of {sorted(PRICE_SOURCES)}")
    return ModelPrice(model_key=str(key), effective_from=effective,
                      input_per_1m_tokens=_number(raw, "input_per_1m_tokens", where),
                      output_per_1m_tokens=_number(raw, "output_per_1m_tokens", where),
                      price_source=source, currency=currency, notes=raw.get("notes"))


def load_rate_card(path: Path | None = None,
                   models: list[ModelEntry] | None = None) -> list[ModelPrice]:
    """Load and validate the rate card; with `models`, every rate must name a catalog model."""
    path = Path(path) if path else PRICING_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read rate card {path}") from exc
    currency = data.get("currency")
    if not isinstance(currency, str) or len(currency) != 3:
        raise ConfigurationError(f"{path.name}: 'currency' must be a 3-letter code")
    known = {m.key for m in models} if models is not None else None
    prices = [_price(raw, i, currency.upper(), known) for i, raw in enumerate(data.get("rates") or [])]
    if not prices:
        raise ConfigurationError(f"{path.name} lists no rates")
    seen = [(p.model_key, p.effective_from) for p in prices]
    duplicates = sorted({s for s in seen if seen.count(s) > 1})
    if duplicates:
        raise ConfigurationError(f"Duplicate model_key/effective_from in {path.name}: {duplicates}")
    return prices


class RateCardSync:
    """Mirrors the YAML rate card into <mkt>.model_price with one MERGE."""

    def __init__(self, names: PlatformNames, logger: logging.Logger):
        self._table = names.table("model_price")
        self._logger = logger

    def statement(self, prices: list[ModelPrice]) -> str:
        rows = ",\n  ".join("(" + ", ".join([
            literal(p.model_key), f"DATE {literal(p.effective_from)}", literal(p.currency),
            f"CAST({p.input_per_1m_tokens!r} AS DOUBLE)", f"CAST({p.output_per_1m_tokens!r} AS DOUBLE)",
            literal(p.price_source), literal(p.notes)]) + ")" for p in prices)
        source = ("SELECT * FROM VALUES " + rows + " AS s(model_key, effective_from, currency, "
                  "input_usd_per_1m_tokens, output_usd_per_1m_tokens, price_source, notes)")
        changed = " OR ".join(f"NOT (t.{c} <=> s.{c})" for c in (
            "currency", "input_usd_per_1m_tokens", "output_usd_per_1m_tokens", "price_source", "notes"))
        return (f"MERGE INTO {self._table} t USING ({source}) s "
                "ON t.model_key = s.model_key AND t.effective_from = s.effective_from "
                f"WHEN MATCHED AND ({changed}) THEN UPDATE SET currency = s.currency, "
                "input_usd_per_1m_tokens = s.input_usd_per_1m_tokens, "
                "output_usd_per_1m_tokens = s.output_usd_per_1m_tokens, "
                "price_source = s.price_source, notes = s.notes, "
                "synced_at = current_timestamp(), synced_by = current_user() "
                "WHEN NOT MATCHED THEN INSERT (model_key, effective_from, currency, "
                "input_usd_per_1m_tokens, output_usd_per_1m_tokens, price_source, notes, synced_at, "
                "synced_by) VALUES (s.model_key, s.effective_from, s.currency, "
                "s.input_usd_per_1m_tokens, s.output_usd_per_1m_tokens, s.price_source, s.notes, "
                "current_timestamp(), current_user()) "
                "WHEN NOT MATCHED BY SOURCE THEN DELETE")

    def sync(self, engine: Any, prices: list[ModelPrice]) -> int:
        from sqlalchemy import text
        with engine.connect() as connection:
            result = connection.execute(text(self.statement(prices).replace(":", "\\:")))
        changed = max(int(getattr(result, "rowcount", 0) or 0), 0)
        self._logger.info("rate_card_synced", extra={
            "table": self._table, "rates": len(prices), "rows_changed": changed,
            "illustrative": sum(p.price_source == "illustrative" for p in prices),
            "kpi_name": "rate_card_rows_changed", "kpi_value": changed})
        return changed
