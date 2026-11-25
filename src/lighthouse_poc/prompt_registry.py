"""MLflow Prompt Registry adapter."""
import logging
from dataclasses import dataclass

import mlflow

from .exceptions import PromptRegistryError


@dataclass(frozen=True)
class PromptReference:
    name: str
    version: str
    uri: str


def _reference(name: str, version) -> PromptReference:
    return PromptReference(name, str(version), f"prompts:/{name}/{version}")


class MLflowPromptRegistry:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def register(self, name: str, template: str, change_note: str, tags=None) -> PromptReference:
        try:
            prompt = mlflow.genai.register_prompt(name=name, template=template,
                                                 commit_message=change_note, tags=tags or {})
            result = _reference(name, prompt.version)
            self._logger.info("prompt_registered", extra={"prompt_name": name, "prompt_version": result.version})
            return result
        except Exception as exc:
            self._logger.exception("prompt_registration_failed", extra={"prompt_name": name})
            raise PromptRegistryError(f"Could not register {name}") from exc

    def latest(self, name: str):
        """Latest registered version, or None when the prompt does not exist yet."""
        try:
            return mlflow.genai.load_prompt(name_or_uri=f"prompts:/{name}@latest")
        except Exception as exc:
            from mlflow.exceptions import MlflowException
            # Local tracking servers report RESOURCE_DOES_NOT_EXIST; Unity Catalog may use NOT_FOUND.
            if (isinstance(exc, MlflowException)
                    and exc.error_code in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}):
                return None
            self._logger.exception("prompt_latest_lookup_failed", extra={"prompt_name": name})
            raise PromptRegistryError(f"Could not look up the latest version of {name}") from exc

    def register_if_changed(self, name: str, template: str, change_note: str,
                            tags=None) -> tuple[PromptReference, bool]:
        """Register a new version only when the template differs from the latest one.

        Versions are immutable, so repeated CI deploys of an unchanged template reuse the
        existing version instead of creating identical versions.
        """
        current = self.latest(name)
        if current is not None and current.template == template:
            ref = _reference(name, current.version)
            self._logger.info("prompt_unchanged", extra={"prompt_name": name,
                                                         "prompt_version": ref.version})
            return ref, False
        return self.register(name, template, change_note, tags), True

    def load(self, name: str, version: str):
        try:
            prompt = mlflow.genai.load_prompt(name_or_uri=f"prompts:/{name}/{version}")
            self._logger.info("prompt_loaded", extra={"prompt_name": name, "prompt_version": version})
            return prompt
        except Exception as exc:
            self._logger.exception("prompt_load_failed", extra={"prompt_name": name, "prompt_version": version})
            raise PromptRegistryError(f"Could not load {name} version {version}") from exc
