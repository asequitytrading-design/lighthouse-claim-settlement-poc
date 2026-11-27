"""Synthetic claims data and the claims copilot's tool function

Revision ID: 0005_claims_data
Revises: 0004_gateway_usage_copy
Create Date: 2026-11-27

Schema <prefix>ai_claims stands in for the client's claims lakehouse (synthetic data only):
  claim_case      one row per claim (40: the six golden cases + 34 generated, deterministic)
  claim_document  documents per claim: received | missing | conflicting
  get_claim_case  SQL function returning one case file as JSON. Databricks' managed MCP server
                  for UC functions (/api/2.0/mcp/functions/<catalog>/<schema>) exposes every
                  function in this schema as an agent tool, so the schema holds tools only.
The data migration (MERGE from lighthouse_poc.agents.claims_copilot.synthetic_claims) keeps the
rows identical to the generator: re-running updates changed rows and removes extra ones.
Column comments describe the data for Genie.
"""
from lighthouse_poc.agents.claims_copilot.synthetic_claims import generate
from lighthouse_poc.platform.migrations import run_sql
from lighthouse_poc.platform.sql import literal

revision = "0005_claims_data"
down_revision = "0004_gateway_usage_copy"
branch_labels = None
depends_on = None

DELTA = ("TBLPROPERTIES ('delta.columnMapping.mode' = 'name', "
         "'delta.enableChangeDataFeed' = 'true')")

CASE_COLUMNS = ["case_id", "product_line", "claim_status", "received_date", "last_update_date",
                "region", "channel", "assigned_team", "claimant_ref", "case_summary", "synthetic"]
DOC_COLUMNS = ["case_id", "document_type", "doc_status", "received_date", "note"]


def _values(rows, columns) -> str:
    return ",\n  ".join("(" + ", ".join(literal(getattr(r, c)) for c in columns) + ")" for r in rows)


def _merge(table: str, rows, columns, keys) -> str:
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    updates = ", ".join(f"t.{c} = s.{c}" for c in columns if c not in keys)
    return (f"MERGE INTO {table} t USING (SELECT * FROM VALUES\n  {_values(rows, columns)}\n"
            f"  AS v({', '.join(columns)})) s ON {on}\n"
            f"WHEN MATCHED THEN UPDATE SET {updates}\n"
            f"WHEN NOT MATCHED THEN INSERT ({', '.join(columns)}) VALUES ({', '.join('s.' + c for c in columns)})\n"
            f"WHEN NOT MATCHED BY SOURCE THEN DELETE")


GET_CLAIM_CASE = """CREATE OR REPLACE FUNCTION {clm}.get_claim_case(
    case_id STRING COMMENT 'Synthetic claim case ID, for example CSET-LTD-002')
RETURNS STRING
COMMENT 'Returns one synthetic claim case file as JSON: product line, status, dates, a short case summary and every document with its status (received, missing or conflicting). Administrative facts only: it never contains coverage, eligibility, fraud or benefit decisions.'
RETURN coalesce(
  (SELECT to_json(named_struct(
      'case_id', first(c.case_id), 'product_line', first(c.product_line),
      'claim_status', first(c.claim_status),
      'received_date', CAST(first(c.received_date) AS STRING),
      'last_update_date', CAST(first(c.last_update_date) AS STRING),
      'region', first(c.region), 'channel', first(c.channel),
      'assigned_team', first(c.assigned_team), 'case_summary', first(c.case_summary),
      'documents', collect_list(named_struct(
          'document_type', d.document_type, 'status', d.doc_status,
          'received_date', CAST(d.received_date AS STRING), 'note', d.note)),
      'synthetic', first(c.synthetic)))
   FROM {clm}.claim_case c
   LEFT JOIN {clm}.claim_document d ON d.case_id = c.case_id
   WHERE c.case_id = upper(trim(get_claim_case.case_id))
   HAVING count(c.case_id) > 0),
  to_json(named_struct('error', 'case not found', 'case_id', get_claim_case.case_id)))"""


def upgrade() -> None:
    cases, docs = generate()
    run_sql(
        "CREATE SCHEMA IF NOT EXISTS {clm} COMMENT 'Synthetic claims data read by marketplace "
        "agents (stand-in for the client claims lakehouse) and the agent tool functions'",
        f"""CREATE TABLE IF NOT EXISTS {{clm}}.claim_case (
            case_id STRING NOT NULL COMMENT 'Synthetic claim case ID, e.g. CSET-LTD-002',
            product_line STRING NOT NULL COMMENT 'std (short-term disability) | ltd (long-term disability) | life | accident | critical_illness | dental',
            claim_status STRING NOT NULL COMMENT 'open | awaiting_documents | ready_for_examiner',
            received_date DATE NOT NULL COMMENT 'Date the claim was received',
            last_update_date DATE COMMENT 'Date of the latest update to the file',
            region STRING COMMENT 'Synthetic region: north | south | east | west',
            channel STRING COMMENT 'How the claim arrived: portal | broker | paper',
            assigned_team STRING COMMENT 'disability | life | health',
            claimant_ref STRING COMMENT 'Synthetic claimant reference (no personal data)',
            case_summary STRING COMMENT 'Short synthetic summary of the file (administrative facts only)',
            synthetic BOOLEAN NOT NULL COMMENT 'Always true: synthetic demo data',
            CONSTRAINT claim_case_pk PRIMARY KEY (case_id)
        ) COMMENT 'Synthetic claim cases (demo stand-in for the client claims lakehouse)' {DELTA}""",
        f"""CREATE TABLE IF NOT EXISTS {{clm}}.claim_document (
            case_id STRING NOT NULL COMMENT 'Claim case ID (claim_case.case_id)',
            document_type STRING NOT NULL COMMENT 'e.g. employer statement, functional capacity evaluation, certified death certificate',
            doc_status STRING NOT NULL COMMENT 'received | missing (expected or referenced, not in the file) | conflicting (details differ from another document)',
            received_date DATE COMMENT 'Date received; NULL when missing',
            note STRING COMMENT 'Short administrative note'
        ) COMMENT 'Documents of each synthetic claim and their status' {DELTA}""",
        _merge("{clm}.claim_case", cases, CASE_COLUMNS, ["case_id"]),
        _merge("{clm}.claim_document", docs, DOC_COLUMNS, ["case_id", "document_type"]),
        GET_CLAIM_CASE,
    )


def downgrade() -> None:
    run_sql("DROP FUNCTION IF EXISTS {clm}.get_claim_case",
            "DROP TABLE IF EXISTS {clm}.claim_document",
            "DROP TABLE IF EXISTS {clm}.claim_case",
            "DROP SCHEMA IF EXISTS {clm}")
