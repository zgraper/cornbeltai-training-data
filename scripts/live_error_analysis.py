#!/usr/bin/env python3
"""Analyze reviewed live routing logs to surface common failure patterns."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

from src.router import DEFAULT_LOG_PATH, load_logged_interactions, resolved_labels

COMPARE_FIELDS = [
    "is_ag_related",
    "needs_rag",
    "needs_web_search",
    "needs_weather_data",
    "needs_farm_data",
    "crops",
    "topics",
    "intent",
    "urgency",
]


def failure_patterns(interactions: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = [row for row in interactions if resolved_labels(row) is not None]
    field_errors: Counter[str] = Counter()
    pattern_errors: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for row in reviewed:
        final_labels = resolved_labels(row)
        prediction = row["prediction"]
        mismatches: list[str] = []
        for field in COMPARE_FIELDS:
            predicted_value = prediction.get(field)
            final_value = final_labels.get(field)
            if isinstance(predicted_value, list):
                equal = sorted(predicted_value) == sorted(final_value)
            else:
                equal = predicted_value == final_value
            if not equal:
                mismatches.append(field)
                field_errors[field] += 1
                confusion[field][f"pred={predicted_value} -> final={final_value}"] += 1
        if mismatches:
            pattern_key = ", ".join(sorted(mismatches))
            pattern_errors[pattern_key] += 1
            examples.append(
                {
                    "interaction_id": row["interaction_id"],
                    "input": row["input"],
                    "mismatches": mismatches,
                    "prediction": prediction,
                    "final_labels": final_labels,
                    "user_feedback": row.get("user_feedback"),
                }
            )

    return {
        "reviewed_examples": len(reviewed),
        "examples_with_errors": len(examples),
        "most_frequent_mistakes": field_errors.most_common(),
        "common_failure_patterns": pattern_errors.most_common(10),
        "top_failure_examples": examples[:10],
        "field_confusions": {field: counter.most_common(5) for field, counter in confusion.items()},
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    interactions = load_logged_interactions(args.log_path)
    report = failure_patterns(interactions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
