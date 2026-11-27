"""Unity Catalog names of the AI Marketplace data platform.

Four schemas per environment, all in one catalog:
  <prefix>ai_marketplace    governed reference data: model catalog, use cases, release decisions,
                            migration history and restore points
  <prefix>ai_observability  usage and quality events: model invocations, evaluations, vector
                            queries, and views joined with system.ai_gateway.usage
  <prefix>ai_knowledge      knowledge base: documents, embedded chunks, vector index
  <prefix>ai_claims         synthetic claims data the agents read (stand-in for the client's
                            claims lakehouse) and the agent tool functions (managed MCP)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..exceptions import ConfigurationError

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_PREFIX = re.compile(r"^([a-z][a-z0-9_]*_)?$")


def dev_prefix(user_name: str) -> str:
    """Personal schema prefix from a user name or e-mail: 'a.b-c@x.com' -> 'dev_a_b_c_'."""
    local = user_name.split("@", 1)[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "_", local).strip("_") or "user"
    return f"dev_{slug}_"


@dataclass(frozen=True)
class PlatformNames:
    catalog: str
    prefix: str = ""

    def __post_init__(self):
        if not _IDENTIFIER.match(self.catalog):
            raise ConfigurationError(f"Invalid catalog name {self.catalog!r}")
        if not _PREFIX.match(self.prefix):
            raise ConfigurationError(
                f"Invalid schema prefix {self.prefix!r}: lowercase letters, digits and _ ending in _")

    # schemas (unqualified, as Alembic and information_schema use them)
    @property
    def marketplace(self) -> str:
        return f"{self.prefix}ai_marketplace"

    @property
    def observability(self) -> str:
        return f"{self.prefix}ai_observability"

    @property
    def knowledge(self) -> str:
        return f"{self.prefix}ai_knowledge"

    @property
    def claims(self) -> str:
        return f"{self.prefix}ai_claims"

    @property
    def schemas(self) -> tuple[str, ...]:
        return (self.marketplace, self.observability, self.knowledge, self.claims)

    def fq(self, schema: str, table: str) -> str:
        """Fully qualified name: catalog.schema.table."""
        return f"{self.catalog}.{schema}.{table}"

    def table(self, name: str) -> str:
        """Fully qualified name of a platform table by its short name."""
        try:
            return self.fq(TABLE_SCHEMAS[name](self), name)
        except KeyError:
            raise ConfigurationError(f"Unknown platform table {name!r}") from None

    def sql_names(self) -> dict[str, str]:
        """Placeholders for migration SQL: {catalog}, {mkt}, {obs}, {kn}, {clm} (qualified schemas)."""
        return {"catalog": self.catalog,
                "mkt": f"{self.catalog}.{self.marketplace}",
                "obs": f"{self.catalog}.{self.observability}",
                "kn": f"{self.catalog}.{self.knowledge}",
                "clm": f"{self.catalog}.{self.claims}"}

    @property
    def vector_index(self) -> str:
        return self.fq(self.knowledge, "document_chunk_index")


TABLE_SCHEMAS = {
    "model_catalog": lambda n: n.marketplace,
    "use_case": lambda n: n.marketplace,
    "release_decision": lambda n: n.marketplace,
    "schema_migration_audit": lambda n: n.marketplace,
    "model_price": lambda n: n.marketplace,
    "model_invocation": lambda n: n.observability,
    "eval_run": lambda n: n.observability,
    "eval_case_result": lambda n: n.observability,
    "vector_query": lambda n: n.observability,
    "gateway_usage": lambda n: n.observability,
    "source_document": lambda n: n.knowledge,
    "document_chunk": lambda n: n.knowledge,
    "claim_case": lambda n: n.claims,
    "claim_document": lambda n: n.claims,
}
