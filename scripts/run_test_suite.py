#!/usr/bin/env python3
"""Run a curated real-world routing test suite against the baseline inference pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

from src.router import DEFAULT_MODEL_DIR, predict_query

TEST_CASES = [
    {
        "name": "short_query_corn_yellow",
        "input": "corn yellow",
        "expected": {
            "is_ag_related": True,
            "crops": ["corn"],
            "topics": ["nutrient"],
            "needs_rag": True,
            "needs_weather_data": False,
            "needs_farm_data": False,
        },
    },
    {
        "name": "vague_field_observation",
        "input": "field looks bad",
        "expected": {
            "is_ag_related": True,
            "crops": ["unknown"],
            "needs_rag": False,
            "needs_farm_data": True,
        },
    },
    {
        "name": "mixed_market_and_planting",
        "input": "price of corn and when should I plant?",
        "expected": {
            "is_ag_related": True,
            "crops": ["corn"],
            "topics": ["management", "market_economics"],
            "needs_web_search": True,
            "needs_weather_data": True,
        },
    },
    {
        "name": "non_ag_noise",
        "input": "what time does the movie start tonight",
        "expected": {
            "is_ag_related": False,
            "needs_rag": False,
            "needs_web_search": False,
            "needs_weather_data": False,
            "needs_farm_data": False,
        },
    },
    {
        "name": "weather_driven_spray_timing",
        "input": "Should I spray before the rain tomorrow?",
        "expected": {
            "is_ag_related": True,
            "topics": ["management", "weather"],
            "needs_rag": True,
            "needs_weather_data": True,
            "needs_farm_data": True,
        },
    },
    {
        "name": "farm_specific_context",
        "input": "Given this field's planting date and hybrid, should I sidedress this week?",
        "expected": {
            "is_ag_related": True,
            "topics": ["management", "nutrient"],
            "needs_rag": True,
            "needs_weather_data": True,
            "needs_farm_data": True,
        },
    },
]


def compare_expected(prediction: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key, expected_value in expected.items():
        actual_value = prediction.get(key)
        if isinstance(expected_value, list):
            actual_set = set(actual_value)
            expected_set = set(expected_value)
            checks[key] = expected_set.issubset(actual_set)
        else:
            checks[key] = actual_value == expected_value
    checks["all_passed"] = all(checks.values())
    return checks



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    results = []
    for case in TEST_CASES:
        result = predict_query(case["input"], model_dir=args.model_dir)
        checks = compare_expected(result["prediction"], case["expected"])
        results.append(
            {
                "name": case["name"],
                "input": case["input"],
                "expected": case["expected"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "checks": checks,
            }
        )

    passed = sum(1 for row in results if row["checks"]["all_passed"])
    summary = {"passed": passed, "total": len(results), "results": results}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
