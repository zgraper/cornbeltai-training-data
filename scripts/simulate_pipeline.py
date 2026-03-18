#!/usr/bin/env python3
"""Simulate downstream module activation from a routing prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router import DEFAULT_MODEL_DIR, predict_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Query to route.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    text = " ".join(args.query).strip()
    if not text:
        text = input("Enter a routing query: ").strip()
    if not text:
        raise SystemExit("No query provided.")

    result = predict_query(text, model_dir=args.model_dir)
    prediction = result["prediction"]
    print(json.dumps({"input": result["input"], "prediction": prediction}, indent=2))
    print()

    if prediction["needs_rag"]:
        print("RAG would be triggered")
    if prediction["needs_web_search"]:
        print("Web search would be triggered")
    if prediction["needs_weather_data"]:
        print("Weather API would be called")
    if prediction["needs_farm_data"]:
        print("Farm data lookup required")
    if not any(prediction[flag] for flag in ["needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]):
        print("No downstream modules would be triggered")


if __name__ == "__main__":
    main()
