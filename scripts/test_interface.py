#!/usr/bin/env python3
"""Interactive CLI for testing live routing predictions and collecting logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router import DEFAULT_LOG_PATH, DEFAULT_MODEL_DIR, log_prediction, predict_query

EXIT_COMMANDS = {":q", ":quit", ":exit", "quit", "exit"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    print("CornbeltAI routing test interface")
    print("Type a query and press Enter. Type :quit to exit.\n")

    while True:
        try:
            text = input("query> ").strip()
        except EOFError:
            print()
            break
        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            break

        result = predict_query(text, model_dir=args.model_dir)
        record = log_prediction(result, log_path=args.log_path)
        payload = {
            "interaction_id": record["interaction_id"],
            "input": result["input"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
        }
        print(json.dumps(payload, indent=2))
        print()


if __name__ == "__main__":
    main()
