#!/usr/bin/env python3
"""Run routing inference for a single query and optionally append the interaction to the feedback log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router import DEFAULT_LOG_PATH, DEFAULT_MODEL_DIR, log_prediction, predict_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="User query text. If omitted, the script prompts on stdin.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing baseline model artifacts.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH, help="Append-only routing interaction log path.")
    parser.add_argument("--no-log", action="store_true", help="Skip writing the inference event to the log file.")
    args = parser.parse_args()

    text = " ".join(args.query).strip()
    if not text:
        text = input("Enter a routing query: ").strip()
    if not text:
        raise SystemExit("No query provided.")

    result = predict_query(text, model_dir=args.model_dir)
    payload = {"input": result["input"], "prediction": result["prediction"], "confidence": result["confidence"]}
    print(json.dumps(payload, indent=2))

    if not args.no_log:
        record = log_prediction(result, log_path=args.log_path)
        print(f"\nLogged interaction: {record['interaction_id']}")


if __name__ == "__main__":
    main()
