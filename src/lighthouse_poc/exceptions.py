"""Typed application failures."""


class LighthouseError(Exception):
    """Base failure."""


class ConfigurationError(LighthouseError):
    """Environment profile is missing, malformed or references unset variables."""


class PromptRegistryError(LighthouseError):
    """Prompt registry failed."""


class ModelInvocationError(LighthouseError):
    """Model gateway call failed.

    ``category`` is a stable, log-safe label (auth, permission, not_found, rate_limit,
    timeout, connection, bad_request, upstream, empty_response, client_init, unknown).
    """

    def __init__(self, message: str, category: str = "unknown"):
        super().__init__(message)
        self.category = category


class OutputValidationError(LighthouseError):
    """Output contract failed."""


class EvaluationError(LighthouseError):
    """Evaluation failed."""
