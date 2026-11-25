"""Register the bundled synthetic claim prompt in the configured MLflow tracking server."""
import argparse
from pathlib import Path

import mlflow

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="claim_settlement_review")
    parser.add_argument("--change-note", default="Initial synthetic claims prompt")
    args = parser.parse_args()
    template = (PROJECT / "prompts" / "claim_settlement_v1.txt").read_text(encoding="utf-8")
    prompt = mlflow.genai.register_prompt(
        name=args.name,
        template=template,
        commit_message=args.change_note,
        tags={"use_case": "claim_settlement_preparation", "synthetic_data": "true"},
    )
    print(f"Registered prompt {args.name} version {prompt.version}")
    print(f"URI: prompts:/{args.name}/{prompt.version}")


if __name__ == "__main__":
    main()
