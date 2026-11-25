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

class MLflowPromptRegistry:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def register(self, name: str, template: str, change_note: str, tags=None) -> PromptReference:
        try:
            prompt = mlflow.genai.register_prompt(name=name, template=template,
                                                 commit_message=change_note, tags=tags or {})
            result = PromptReference(name, str(prompt.version), f"prompts:/{name}/{prompt.version}")
            self._logger.info("prompt_registered", extra={"prompt_name": name, "prompt_version": result.version})
            return result
        except Exception as exc:
            self._logger.exception("prompt_registration_failed", extra={"prompt_name": name})
            raise PromptRegistryError(f"Could not register {name}") from exc

    def load(self, name: str, version: str):
        try:
            prompt = mlflow.genai.load_prompt(name_or_uri=f"prompts:/{name}/{version}")
            self._logger.info("prompt_loaded", extra={"prompt_name": name, "prompt_version": version})
            return prompt
        except Exception as exc:
            self._logger.exception("prompt_load_failed", extra={"prompt_name": name, "prompt_version": version})
            raise PromptRegistryError(f"Could not load {name} version {version}") from exc

