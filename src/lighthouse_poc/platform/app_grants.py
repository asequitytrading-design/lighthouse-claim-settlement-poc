"""Unity Catalog grants for an application identity (the demo app's service principal).

Least privilege per schema (`lighthouse platform app grants --principal <application id>`):
  <prefix>ai_claims         USE SCHEMA, SELECT, EXECUTE   case data + the get_claim_case tool
  <prefix>ai_knowledge      USE SCHEMA, SELECT            knowledge chunks and the vector index
  <prefix>ai_observability  USE SCHEMA, SELECT, MODIFY    read usage/eval views, write call events
  <prefix>ai_marketplace    USE SCHEMA, SELECT            model catalog, prices, schema revision
  <prompt schema>           USE SCHEMA, EXECUTE           load the approved prompt (@champion)
Re-runnable (GRANT is idempotent). The SQL warehouse and the Genie space are granted through the
app's resources (bundle), not here.
"""
from __future__ import annotations

import re

from ..exceptions import ConfigurationError
from .naming import PlatformNames

_PRINCIPAL = re.compile(r"^[A-Za-z0-9@._-]+$")


def grant_statements(names: PlatformNames, principal: str, prompt_schema: str | None = None) -> list[str]:
    if not _PRINCIPAL.match(principal or ""):
        raise ConfigurationError(f"Invalid principal {principal!r}")
    who = f"`{principal}`"
    statements = [f"GRANT USE CATALOG ON CATALOG {names.catalog} TO {who}"]
    for schema, privileges in ((names.claims, "USE SCHEMA, SELECT, EXECUTE"),
                               (names.knowledge, "USE SCHEMA, SELECT"),
                               (names.observability, "USE SCHEMA, SELECT, MODIFY"),
                               (names.marketplace, "USE SCHEMA, SELECT")):
        statements.append(f"GRANT {privileges} ON SCHEMA {names.catalog}.{schema} TO {who}")
    if prompt_schema:
        statements.append(f"GRANT USE SCHEMA, EXECUTE ON SCHEMA {names.catalog}.{prompt_schema} TO {who}")
    return statements
