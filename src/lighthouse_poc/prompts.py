"""Prompt references for the developer workflow: registered versions, aliases and drafts.

A reference is one of:
  ``3``              registered version 3
  ``@champion``      the version an alias points to
  ``latest``         highest registered version
  ``file:<path>``    an unregistered draft template on disk

Drafts let developers iterate without creating registry versions. A draft is labelled by
the hash of its content (``draft-<sha>``) so every trace and run says exactly which text
ran; only registered versions can be approved and published.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from mlflow.entities.model_registry import PromptVersion

from .exceptions import ConfigurationError

_ALIAS = re.compile(r"^@[A-Za-z][A-Za-z0-9_-]*$")
REF_HELP = "prompt reference: <version> | @<alias> | latest | file:<path>"


@dataclass(frozen=True)
class PromptRef:
    kind: str   # version | alias | latest | file
    value: str

    @classmethod
    def parse(cls, text: str) -> "PromptRef":
        text = (text or "").strip()
        if text.startswith("file:") and len(text) > 5:
            return cls("file", text[5:])
        if text == "latest":
            return cls("latest", text)
        if _ALIAS.match(text):
            return cls("alias", text[1:])
        if text.isdigit() and int(text) > 0:
            return cls("version", str(int(text)))
        raise ConfigurationError(f"Invalid prompt reference {text!r}; expected {REF_HELP}")


@dataclass(frozen=True)
class ResolvedPrompt:
    """A prompt template plus the lineage needed to label traces and runs."""
    name: str
    label: str                     # registered version number or draft-<sha>
    template: str
    source: str                    # registry | file
    reference: str                 # what the developer asked for, e.g. @champion
    aliases: tuple[str, ...] = ()
    commit_message: str | None = None
    creation_timestamp: int | None = None
    path: str | None = None
    tags: dict = field(default_factory=dict)

    @property
    def is_draft(self) -> bool:
        return self.source == "file"

    def format(self, **variables) -> str:
        """Same rendering as a registered version, so drafts behave identically."""
        return PromptVersion(name=self.name, version=1, template=self.template).format(**variables)


def draft_label(template: str) -> str:
    return "draft-" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:8]


class PromptResolver:
    """Turns a PromptRef into a ResolvedPrompt using the registry or the filesystem."""

    def __init__(self, registry, name: str, base_dir: Path | None = None):
        self._registry, self._name = registry, name
        self._base_dir = base_dir or Path.cwd()

    def resolve(self, ref: PromptRef | str) -> ResolvedPrompt:
        ref = PromptRef.parse(ref) if isinstance(ref, str) else ref
        if ref.kind == "file":
            path = Path(ref.value).expanduser()
            path = path if path.is_absolute() else self._base_dir / path
            try:
                template = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigurationError(f"Cannot read draft prompt {path}") from exc
            if not template.strip():
                raise ConfigurationError(f"Draft prompt {path} is empty")
            return ResolvedPrompt(self._name, draft_label(template), template, "file",
                                  f"file:{ref.value}", path=str(path))
        if ref.kind == "latest":
            prompt = self._registry.latest(self._name)
            if prompt is None:
                raise ConfigurationError(f"Prompt {self._name} has no registered versions")
        elif ref.kind == "alias":
            prompt = self._registry.load_alias(self._name, ref.value)
        else:
            prompt = self._registry.load(self._name, ref.value)
        reference = {"alias": f"@{ref.value}"}.get(ref.kind, ref.value)
        return from_prompt_version(prompt, self._name, reference)


def from_prompt_version(prompt, name: str, reference: str) -> ResolvedPrompt:
    return ResolvedPrompt(
        name=name, label=str(prompt.version), template=prompt.template, source="registry",
        reference=reference, aliases=tuple(getattr(prompt, "aliases", None) or ()),
        commit_message=getattr(prompt, "commit_message", None),
        creation_timestamp=getattr(prompt, "creation_timestamp", None),
        tags=dict(getattr(prompt, "tags", None) or {}))
