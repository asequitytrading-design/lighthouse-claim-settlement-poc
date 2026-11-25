"""Typed application failures."""
class LighthouseError(Exception): """Base failure."""
class PromptRegistryError(LighthouseError): """Prompt registry failed."""
class ModelInvocationError(LighthouseError): """Model endpoint failed."""
class OutputValidationError(LighthouseError): """Output contract failed."""
class EvaluationError(LighthouseError): """Evaluation failed."""

