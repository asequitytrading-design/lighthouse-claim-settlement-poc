"""Register the bundled synthetic claim prompt in the profile's MLflow prompt registry."""
import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import mlflow  # noqa: E402

from lighthouse_poc.config import load_dotenv, load_profile, require  # noqa: E402
from lighthouse_poc.exceptions import LighthouseError  # noqa: E402
from lighthouse_poc.logging_config import configure_logging  # noqa: E402
from lighthouse_poc.prompt_registry import MLflowPromptRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Profile name or YAML path (default: $LIGHTHOUSE_PROFILE or local)")
    parser.add_argument("--name", help="Override the profile's prompt name")
    parser.add_argument("--template", default=str(PROJECT / "prompts" / "claim_settlement_v1.txt"))
    parser.add_argument("--change-note", default="Initial synthetic claims prompt")
    args = parser.parse_args()
    logger = configure_logging()
    load_dotenv()
    try:
        profile = load_profile(args.profile)
        mlflow.set_tracking_uri(require(profile.tracking_uri, "mlflow.tracking_uri"))
        name = args.name or require(profile.prompt_name, "registry.prompt_name")
        template = Path(args.template).read_text(encoding="utf-8")
        ref = MLflowPromptRegistry(logger).register(
            name, template, args.change_note,
            {"use_case": "claim_settlement_preparation", "synthetic_data": "true",
             "non_decisional": "true", "profile": profile.name})
    except (LighthouseError, OSError) as exc:
        logger.error("prompt_registration_script_failed", exc_info=True,
                     extra={"error_type": type(exc).__name__})
        return 1
    print(f"Registered prompt {ref.name} version {ref.version}\nURI: {ref.uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
