"""Genie space over the synthetic claims tables (the copilot's portfolio-question tool).

Created idempotently by title (`lighthouse platform genie ensure`): one space per environment
prefix, over <prefix>ai_claims.claim_case and claim_document, with instructions and sample
questions. The agent reaches it through Databricks' managed Genie MCP server
(/api/2.0/mcp/genie/<space_id>). Synthetic data only.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from ..exceptions import ConfigurationError
from .naming import PlatformNames

INSTRUCTIONS = [
    "* Data is synthetic demo data for claims administration; never infer real people.\n",
    "* product_line values: std = short-term disability, ltd = long-term disability, life, accident, "
    "critical_illness, dental.\n",
    "* A document is missing when claim_document.doc_status = 'missing'; conflicting when 'conflicting'.\n",
    "* Answer with counts, lists and dates only. Never decide coverage, eligibility, fraud or benefit amounts.\n",
]
SAMPLE_QUESTIONS = [
    "How many long-term disability claims are missing a functional capacity evaluation?",
    "Which open claims are awaiting documents, by product line?",
    "List the claims with conflicting documents",
]


@dataclass(frozen=True)
class GenieSpace:
    space_id: str
    title: str
    created: bool


def space_title(names: PlatformNames) -> str:
    return f"Lighthouse claims ({names.prefix.rstrip('_') or 'shared'})"


def serialized_space(names: PlatformNames) -> str:
    def ident() -> str:
        return uuid.uuid4().hex
    tables = sorted(names.fq(names.claims, t) for t in ("claim_case", "claim_document"))
    missing_fce = (
        "SELECT count(DISTINCT c.case_id) AS claims\n"
        f"FROM {names.fq(names.claims, 'claim_case')} c\n"
        f"JOIN {names.fq(names.claims, 'claim_document')} d ON d.case_id = c.case_id\n"
        "WHERE c.product_line = 'ltd' AND d.document_type = 'functional capacity evaluation'\n"
        "  AND d.doc_status = 'missing'")
    space = {
        "version": 2,
        "config": {"sample_questions": [{"id": ident(), "question": [q]} for q in SAMPLE_QUESTIONS]},
        "data_sources": {"tables": [{"identifier": t} for t in tables]},
        "instructions": {
            "text_instructions": [{"id": ident(), "content": INSTRUCTIONS}],
            "example_question_sqls": [{"id": ident(), "question": [SAMPLE_QUESTIONS[0]],
                                       "sql": [line + "\n" for line in missing_fce.splitlines()]}],
        },
    }
    return json.dumps(space)


class GenieSpaces:
    def __init__(self, client: Any, logger: logging.Logger):
        self._client, self._logger = client, logger      # databricks.sdk WorkspaceClient

    def find(self, title: str) -> str | None:
        """The oldest space whose title starts with `title` (Genie suffixes duplicate titles
        with a timestamp, and its list is eventually consistent)."""
        matches, token = [], None
        while True:
            response = self._client.genie.list_spaces(page_token=token) if token else \
                self._client.genie.list_spaces()
            matches += [s for s in (getattr(response, "spaces", None) or [])
                        if (s.title or "").startswith(title)]
            token = getattr(response, "next_page_token", None)
            if not token:
                break
        if not matches:
            return None
        matches.sort(key=lambda s: (getattr(s, "create_time", None) or "", s.space_id))
        return matches[0].space_id

    def ensure(self, names: PlatformNames, warehouse_id: str,
               configured_id: str | None = None) -> GenieSpace:
        """The configured space (platform.genie_space_id), else the one found by title, else new."""
        title = space_title(names)
        if configured_id:
            return GenieSpace(configured_id, title, False)
        existing = self.find(title)
        if existing:
            return GenieSpace(existing, title, False)
        if not warehouse_id:
            raise ConfigurationError("A SQL warehouse ID is required to create the Genie space")
        space = self._client.genie.create_space(
            warehouse_id=warehouse_id, serialized_space=serialized_space(names), title=title,
            description="Synthetic claim cases and documents for the Lighthouse claims copilot demo.")
        self._logger.info("genie_space_created", extra={"space_id": space.space_id, "title": title})
        return GenieSpace(space.space_id, title, True)
