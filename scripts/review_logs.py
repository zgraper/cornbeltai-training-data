#!/usr/bin/env python3
"""Manually review logged routing interactions and append review decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

from src.router import DEFAULT_LOG_PATH, append_review, load_logged_interactions

LIST_FIELDS = ["crops", "topics"]
BOOL_FIELDS = ["is_ag_related", "needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]
TEXT_FIELDS = ["intent", "urgency"]


def prompt_bool(label: str, current: bool) -> bool:
    raw = input(f"  {label} [{str(current).lower()}]: ").strip().lower()
    if not raw:
        return current
    return raw in {"1", "true", "t", "yes", "y"}



def prompt_list(label: str, current: list[str]) -> list[str]:
    raw = input(f"  {label} [{', '.join(current)}]: ").strip()
    if not raw:
        return current
    return [value.strip() for value in raw.split(",") if value.strip()]



def prompt_text(label: str, current: str) -> str:
    raw = input(f"  {label} [{current}]: ").strip()
    return raw or current



def prompt_corrected_labels(current: dict[str, Any]) -> dict[str, Any]:
    corrected = dict(current)
    print("Enter corrected labels. Press Enter to keep the current value.")
    for field in BOOL_FIELDS:
        corrected[field] = prompt_bool(field, bool(corrected[field]))
    for field in LIST_FIELDS:
        corrected[field] = prompt_list(field, list(corrected[field]))
    for field in TEXT_FIELDS:
        corrected[field] = prompt_text(field, str(corrected[field]))
    return corrected



def show_interaction(index: int, total: int, row: dict[str, Any]) -> None:
    print(f"\n[{index}/{total}] interaction_id={row['interaction_id']} timestamp={row['timestamp']}")
    print(f"input: {row['input']}")
    print("prediction:")
    print(json.dumps(row["prediction"], indent=2))
    print("confidence:")
    print(json.dumps(row["confidence"], indent=2))
    print(f"review status: {row.get('user_feedback') or 'unreviewed'}")
    if row.get("corrected_labels"):
        print("current corrected labels:")
        print(json.dumps(row["corrected_labels"], indent=2))



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--all", action="store_true", help="Review all interactions instead of only unreviewed ones.")
    args = parser.parse_args()

    interactions = load_logged_interactions(args.log_path)
    if not args.all:
        interactions = [row for row in interactions if row.get("user_feedback") is None]
    if not interactions:
        print("No interactions available for review.")
        return

    total = len(interactions)
    for index, row in enumerate(interactions, start=1):
        show_interaction(index, total, row)
        action = input("Mark as [c]orrect, [i]ncorrect/edit, [s]kip, [q]uit: ").strip().lower()
        if action in {"q", "quit"}:
            break
        if action in {"s", "skip", ""}:
            continue
        if action in {"c", "correct"}:
            append_review(row["interaction_id"], user_feedback="correct", corrected_labels=None, log_path=args.log_path)
            print("Saved review: correct")
            continue
        if action in {"i", "incorrect", "edit"}:
            corrected = prompt_corrected_labels(row["prediction"])
            notes = input("Optional review notes: ").strip() or None
            append_review(
                row["interaction_id"],
                user_feedback="incorrect",
                corrected_labels=corrected,
                review_notes=notes,
                log_path=args.log_path,
            )
            print("Saved review: incorrect with corrected labels")
            continue
        print("Unknown action, skipping.")


if __name__ == "__main__":
    main()
