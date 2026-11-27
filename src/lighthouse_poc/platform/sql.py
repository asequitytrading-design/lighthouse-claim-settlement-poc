"""SQL literal rendering for generated Databricks SQL statements.

Used for trusted, repository-controlled data (the model catalog, synthetic knowledge
documents) where a bulk VALUES list is clearer than hundreds of bind parameters. Never pass
user-supplied text through these helpers into ad-hoc statements.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Iterable


def literal(value: Any) -> str:
    """Render a Python value as a Databricks SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("NaN/inf cannot be stored as a SQL literal")
        return repr(value)
    if isinstance(value, datetime):
        return f"TIMESTAMP {literal(value.isoformat())}"
    if isinstance(value, date):
        return f"DATE {literal(value.isoformat())}"
    if isinstance(value, (list, tuple)):
        return "array(" + ", ".join(literal(v) for v in value) + ")" if value else "array()"
    text = str(value)
    # Databricks string literals: backslash escapes; escape backslash first, then quote.
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def float_array(values: Iterable[float]) -> str:
    """ARRAY<FLOAT> literal for an embedding vector."""
    return "CAST(array(" + ", ".join(repr(float(v)) for v in values) + ") AS ARRAY<FLOAT>)"
