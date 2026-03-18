#!/usr/bin/env python3
"""Generate a rich report for the CornbeltAI routing dataset."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "routing"
SPLITS = ["train.jsonl", "val.jsonl", "test.jsonl"]
EDGE_COMBOS = {
    "weather_only": (False, False, True, False),
    "web_only": (False, True, False, False),
    "farm_only": (False, False, False, True),
    "rag_plus_weather": (True, False, True, False),
    "rag_plus_farm": (True, False, False, True),
    "web_plus_weather": (False, True, True, False),
}
VAGUE_REGEX = re.compile(
    r"\b(looks bad|looks rough|not right|acting up|field looks|weird|off|something wrong|now what|"
    r"what is this|too wet yet|spray today|beans yellow|crop looks|help)\b"
)


def load_rows() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for split in SPLITS:
        path = DATASET_DIR / split
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append((split, json.loads(line)))
    return rows


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9? ]", "", lowered)
    return lowered


def fmt_counter(counter: Counter, total: int | None = None) -> str:
    lines = []
    for key, value in counter.most_common():
        suffix = f" ({value / total:.1%})" if total else ""
        lines.append(f"  - {key}: {value}{suffix}")
    return "\n".join(lines)


def sample_examples(rows: list[tuple[str, dict]], predicate, limit: int = 3) -> list[str]:
    picks = []
    for split, row in rows:
        if predicate(row):
            picks.append(f"[{split}] {row['id']}: {row['input']}")
            if len(picks) >= limit:
                break
    return picks


def warning_flags(total: int, ag_total: int, non_ag_total: int, topic_counts: Counter, route_combo_counts: Counter, short_queries: int, vague_queries: int, multi_topic: int) -> list[str]:
    warnings = []
    non_ag_ratio = non_ag_total / total
    if not 0.25 <= non_ag_ratio <= 0.35:
        warnings.append(f"non-ag ratio is {non_ag_ratio:.1%}; expected 25-35%")
    if short_queries / total < 0.08:
        warnings.append(f"short query ratio is {short_queries / total:.1%}; expected at least 8%")
    if vague_queries / total < 0.04:
        warnings.append(f"vague query ratio is {vague_queries / total:.1%}; expected at least 4%")
    if multi_topic / total < 0.20:
        warnings.append(f"multi-topic ratio is {multi_topic / total:.1%}; expected at least 20%")
    for combo_name, combo in EDGE_COMBOS.items():
        if route_combo_counts[combo] < 20:
            warnings.append(f"edge combo {combo_name} has only {route_combo_counts[combo]} rows")
    low_topic_threshold = max(20, int(ag_total * 0.04))
    for topic, count in sorted(topic_counts.items()):
        if count < low_topic_threshold:
            warnings.append(f"topic {topic} is light at {count} rows (< {low_topic_threshold})")
    return warnings


def build_report(rows: list[tuple[str, dict]]) -> str:
    total = len(rows)
    split_counts = Counter(split for split, _ in rows)
    ag_total = sum(1 for _, row in rows if row["labels"]["is_ag_related"])
    non_ag_total = total - ag_total
    crop_counts = Counter()
    topic_counts = Counter()
    intent_counts = Counter()
    urgency_counts = Counter()
    difficulty_counts = Counter()
    flag_counts = Counter()
    route_combo_counts = Counter()
    duplicate_inputs = Counter()
    short_queries = 0
    vague_queries = 0
    multi_topic = 0

    for _, row in rows:
        labels = row["labels"]
        difficulty_counts[row["meta"]["difficulty"]] += 1
        intent_counts[labels["intent"]] += 1
        urgency_counts[labels["urgency"]] += 1
        duplicate_inputs[normalize_text(row["input"])] += 1
        for crop in labels["crops"]:
            crop_counts[crop] += 1
        for topic in labels["topics"]:
            topic_counts[topic] += 1
        for flag in ["needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]:
            flag_counts[f"{flag}=true"] += int(labels[flag])
            flag_counts[f"{flag}=false"] += int(not labels[flag])
        combo = (labels["needs_rag"], labels["needs_web_search"], labels["needs_weather_data"], labels["needs_farm_data"])
        route_combo_counts[combo] += 1
        if len(row["input"].split()) <= 5:
            short_queries += 1
        if VAGUE_REGEX.search(row["input"].lower()):
            vague_queries += 1
        if len(labels["topics"]) > 1:
            multi_topic += 1

    duplicates = sum(1 for count in duplicate_inputs.values() if count > 1)
    warnings = warning_flags(total, ag_total, non_ag_total, topic_counts, route_combo_counts, short_queries, vague_queries, multi_topic)
    samples = {
        "non_ag": sample_examples(rows, lambda row: not row["labels"]["is_ag_related"]),
        "weather_only": sample_examples(rows, lambda row: row["labels"]["is_ag_related"] and (row["labels"]["needs_rag"], row["labels"]["needs_web_search"], row["labels"]["needs_weather_data"], row["labels"]["needs_farm_data"]) == EDGE_COMBOS["weather_only"]),
        "web_only": sample_examples(rows, lambda row: row["labels"]["is_ag_related"] and (row["labels"]["needs_rag"], row["labels"]["needs_web_search"], row["labels"]["needs_weather_data"], row["labels"]["needs_farm_data"]) == EDGE_COMBOS["web_only"]),
        "farm_only": sample_examples(rows, lambda row: row["labels"]["is_ag_related"] and (row["labels"]["needs_rag"], row["labels"]["needs_web_search"], row["labels"]["needs_weather_data"], row["labels"]["needs_farm_data"]) == EDGE_COMBOS["farm_only"]),
        "rag_weather": sample_examples(rows, lambda row: row["labels"]["is_ag_related"] and (row["labels"]["needs_rag"], row["labels"]["needs_web_search"], row["labels"]["needs_weather_data"], row["labels"]["needs_farm_data"]) == EDGE_COMBOS["rag_plus_weather"]),
        "hard_adversarial": sample_examples(rows, lambda row: row["meta"]["difficulty"] == "hard" and row["labels"]["is_ag_related"]),
    }

    lines = [
        "CornbeltAI Routing Dataset Report",
        "=" * 32,
        f"Total rows: {total}",
        f"Split sizes: {dict(split_counts)}",
        f"Ag-related: {ag_total} ({ag_total / total:.1%})",
        f"Non-ag: {non_ag_total} ({non_ag_total / total:.1%})",
        f"Short queries (1-5 words): {short_queries} ({short_queries / total:.1%})",
        f"Vague queries: {vague_queries} ({vague_queries / total:.1%})",
        f"Multi-topic queries: {multi_topic} ({multi_topic / total:.1%})",
        f"Normalized duplicate inputs: {duplicates}",
        "",
        "Crop distribution",
        fmt_counter(crop_counts, ag_total),
        "",
        "Topic distribution",
        fmt_counter(topic_counts, ag_total),
        "",
        "Intent distribution",
        fmt_counter(intent_counts, total),
        "",
        "Urgency distribution",
        fmt_counter(urgency_counts, total),
        "",
        "Difficulty distribution",
        fmt_counter(difficulty_counts, total),
        "",
        "Routing flag distribution",
        fmt_counter(flag_counts, total),
        "",
        "Edge routing combinations",
        *[f"  - {name}: {route_combo_counts[combo]}" for name, combo in EDGE_COMBOS.items()],
        "",
        "Sample examples by major class",
    ]
    for label, picks in samples.items():
        lines.append(f"  - {label}:")
        lines.extend(f"      * {pick}" for pick in picks)
    lines.extend(["", "Warning flags"])
    if warnings:
        lines.extend(f"  - {warning}" for warning in warnings)
    else:
        lines.append("  - None")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional output file for the report.")
    args = parser.parse_args()

    report = build_report(load_rows())
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
