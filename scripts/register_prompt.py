"""Register the bundled synthetic claim prompt in the profile's MLflow prompt registry.

  python scripts/register_prompt.py --profile local
  python scripts/register_prompt.py --profile databricks --if-changed \\
      --env LIGHTHOUSE_SCHEMA=lighthouse --task-value-key prompt_version   # job task
"""
import argparse
import sys
from pathlib import Path

try:
    PROJECT = Path(__file__).resolve().parents[1]
except NameError:  # some job runtimes execute scripts without __file__
    PROJECT = Path(sys.argv[0]).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import mlflow  # noqa: E402

from lighthouse_poc.config import (apply_env_overrides, load_dotenv,  # noqa: E402
                                   load_profile, require)
from lighthouse_poc.exceptions import LighthouseError  # noqa: E402
from lighthouse_poc.logging_config import configure_logging  # noqa: E402
from lighthouse_poc.prompt_registry import MLflowPromptRegistry  # noqa: E402
from lighthouse_poc.runtime import load_runtime_profile, publish_task_value  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="Profile name or YAML path (default: $LIGHTHOUSE_PROFILE or local)")
    parser.add_argument("--name", help="Override the profile's prompt name")
    parser.add_argument("--template", default=str(PROJECT / "prompts" / "claim_settlement_v1.txt"))
    parser.add_argument("--change-note", default="Initial synthetic claims prompt")
    parser.add_argument("--if-changed", action="store_true",
                        help="Reuse the latest version when the template is identical")
    parser.add_argument("--env", action="append", metavar="KEY=VALUE",
                        help="Set a non-secret profile input (repeatable)")
    parser.add_argument("--task-value-key",
                        help="Publish the version as this Databricks job task value")
    args = parser.parse_args()
    logger = configure_logging()
    load_dotenv(PROJECT / ".env")
    try:
        apply_env_overrides(args.env)
        profile = load_runtime_profile(args.profile, logger)
        mlflow.set_tracking_uri(require(profile.tracking_uri, "mlflow.tracking_uri"))
        if profile.registry_uri:
            mlflow.set_registry_uri(profile.registry_uri)
        name = args.name or require(profile.prompt_name, "registry.prompt_name")
        template = Path(args.template).read_text(encoding="utf-8")
        tags = {"use_case": "claim_settlement_preparation", "synthetic_data": "true",
                "non_decisional": "true", "profile": profile.name}
        registry = MLflowPromptRegistry(logger)
        if args.if_changed:
            ref, created = registry.register_if_changed(name, template, args.change_note, tags)
        else:
            ref, created = registry.register(name, template, args.change_note, tags), True
        if args.task_value_key:
            publish_task_value(args.task_value_key, ref.version, logger)
    except (LighthouseError, OSError) as exc:
        logger.error("prompt_registration_script_failed", exc_info=True,
                     extra={"error_type": type(exc).__name__})
        return 1
    print(f"{'Registered' if created else 'Unchanged'} prompt {ref.name} version {ref.version}\n"
          f"URI: {ref.uri}")
    return 0


if __name__ == "__main__":
    # Exit only on failure: Databricks serverless runs scripts in an IPython runner where
    # even sys.exit(0) raises SystemExit and marks the task as failed.
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)
