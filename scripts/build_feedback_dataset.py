#!/usr/bin/env python3
"""Build a feedback training split from reviewed routing interactions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router import DEFAULT_LOG_PATH, load_logged_interactions, resolved_labels


def build_feedback_rows(interactions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for interaction in interactions:
        labels = resolved_labels(interaction)
        if labels is None:
            continue
        rows.append(
            {
                "id": f"feedback_{interaction['interaction_id']}",
                "input": interaction["input"],
                "labels": labels,
                "meta": {
                    "source_type": "real_world_feedback",
                    "original_interaction_id": interaction["interaction_id"],
                    "review_count": len(interaction.get("reviews", [])),
                    "feedback": interaction.get("user_feedback"),
                },
            }
        )
    return rows



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--output", type=Path, default=Path("datasets/routing/feedback.jsonl"))
    args = parser.parse_args()

    interactions = load_logged_interactions(args.log_path)
    rows = build_feedback_rows(interactions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} reviewed feedback examples to {args.output}")


if __name__ == "__main__":
    main()
