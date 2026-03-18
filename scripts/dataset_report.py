#!/usr/bin/env python3
"""Print summary statistics for the CornbeltAI routing dataset."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "routing"
SPLITS = ["train.jsonl", "val.jsonl", "test.jsonl"]


def load_rows() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for split in SPLITS:
        path = DATASET_DIR / split
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append((split, json.loads(line)))
    return rows


def fmt_counter(counter: Counter, limit: int | None = None) -> str:
    items = counter.most_common(limit)
    return "\n".join(f"  - {key}: {value}" for key, value in items)


def main() -> None:
    rows = load_rows()
    total = len(rows)
    ag_total = sum(1 for _, row in rows if row["labels"]["is_ag_related"])
    non_ag_total = total - ag_total

    split_counts = Counter(split for split, _ in rows)
    crop_counts = Counter()
    topic_counts = Counter()
    intent_counts = Counter()
    urgency_counts = Counter()
    flag_counts = Counter()
    combo_examples: defaultdict[tuple[bool, bool, bool, bool, bool], list[str]] = defaultdict(list)

    for split, row in rows:
        labels = row["labels"]
        for crop in labels["crops"]:
            crop_counts[crop] += 1
        for topic in labels["topics"]:
            topic_counts[topic] += 1
        intent_counts[labels["intent"]] += 1
        urgency_counts[labels["urgency"]] += 1
        for flag in ["needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]:
            flag_counts[f"{flag}=true"] += int(labels[flag])
            flag_counts[f"{flag}=false"] += int(not labels[flag])
        combo = (
            labels["is_ag_related"],
            labels["needs_rag"],
            labels["needs_web_search"],
            labels["needs_weather_data"],
            labels["needs_farm_data"],
        )
        if len(combo_examples[combo]) < 3:
            combo_examples[combo].append(f"[{split}] {row['id']}: {row['input']}")

    print("CornbeltAI Routing Dataset Report")
    print("=" * 34)
    print(f"Total rows: {total}")
    print(f"Split sizes: {dict(split_counts)}")
    print(f"Ag-related: {ag_total} ({ag_total / total:.1%})")
    print(f"Non-ag: {non_ag_total} ({non_ag_total / total:.1%})")
    print()
    print("Crop frequencies")
    print(fmt_counter(crop_counts))
    print()
    print("Topic frequencies")
    print(fmt_counter(topic_counts))
    print()
    print("Intent frequencies")
    print(fmt_counter(intent_counts))
    print()
    print("Urgency frequencies")
    print(fmt_counter(urgency_counts))
    print()
    print("Routing flag distribution")
    print(fmt_counter(flag_counts))
    print()
    print("Example routing combinations")
    sorted_combos = sorted(combo_examples.items(), key=lambda item: (-len(item[1]), item[0]))
    for combo, samples in sorted_combos[:12]:
        print(f"  - combo={combo}")
        for sample in samples:
            print(f"      * {sample}")


if __name__ == "__main__":
    main()
