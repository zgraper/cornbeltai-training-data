#!/usr/bin/env python3
"""Validate the CornbeltAI routing dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "routing"
SCHEMA_PATH = ROOT / "schema" / "routing_schema.json"
SPLITS = [DATASET_DIR / "train.jsonl", DATASET_DIR / "val.jsonl", DATASET_DIR / "test.jsonl"]

ALLOWED_CROPS = {"corn", "soybean", "both", "unknown"}
ALLOWED_TOPICS = {
    "weather",
    "disease",
    "pest",
    "weed",
    "nutrient",
    "soil",
    "management",
    "market_economics",
    "equipment",
    "ag_technology",
    "policy_regulation",
    "general_agronomy",
}
ALLOWED_INTENTS = {
    "question",
    "diagnosis",
    "recommendation",
    "planning",
    "monitoring",
    "information_lookup",
    "comparison",
    "other",
}
ALLOWED_URGENCY = {"low", "medium", "high"}
REQUIRED_TOP_KEYS = {"id", "input", "labels", "meta"}
REQUIRED_LABEL_KEYS = {
    "is_ag_related",
    "crops",
    "topics",
    "needs_rag",
    "needs_web_search",
    "needs_weather_data",
    "needs_farm_data",
    "intent",
    "urgency",
}
REQUIRED_META_KEYS = {"source_type", "difficulty", "notes"}
ALLOWED_DIFFICULTY = {"simple", "medium", "hard"}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def validate_row(row: dict, source: Path, line_no: int, seen_ids: set[str]) -> None:
    if not isinstance(row, dict):
        fail(f"{source}:{line_no}: row must be a JSON object")
    if set(row.keys()) != REQUIRED_TOP_KEYS:
        fail(f"{source}:{line_no}: top-level keys must be {sorted(REQUIRED_TOP_KEYS)}")

    row_id = row["id"]
    if not isinstance(row_id, str) or not row_id.startswith("route_") or len(row_id) != 12 or not row_id[6:].isdigit():
        fail(f"{source}:{line_no}: invalid id {row_id!r}")
    if row_id in seen_ids:
        fail(f"{source}:{line_no}: duplicate id {row_id}")
    seen_ids.add(row_id)

    if not isinstance(row["input"], str) or not row["input"].strip():
        fail(f"{source}:{line_no}: input must be a non-empty string")

    labels = row["labels"]
    if not isinstance(labels, dict) or set(labels.keys()) != REQUIRED_LABEL_KEYS:
        fail(f"{source}:{line_no}: labels must contain {sorted(REQUIRED_LABEL_KEYS)}")

    meta = row["meta"]
    if not isinstance(meta, dict) or set(meta.keys()) != REQUIRED_META_KEYS:
        fail(f"{source}:{line_no}: meta must contain {sorted(REQUIRED_META_KEYS)}")

    for key in ["is_ag_related", "needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]:
        if not isinstance(labels[key], bool):
            fail(f"{source}:{line_no}: labels.{key} must be boolean")

    crops = labels["crops"]
    if not isinstance(crops, list) or len(crops) != len(set(crops)):
        fail(f"{source}:{line_no}: crops must be a unique list")
    if any(crop not in ALLOWED_CROPS for crop in crops):
        fail(f"{source}:{line_no}: invalid crop value in {crops}")
    if "both" in crops and crops != ["both"]:
        fail(f"{source}:{line_no}: 'both' cannot be combined with other crop labels")
    if "unknown" in crops and crops != ["unknown"]:
        fail(f"{source}:{line_no}: 'unknown' cannot be combined with explicit crop labels")

    topics = labels["topics"]
    if not isinstance(topics, list) or len(topics) != len(set(topics)):
        fail(f"{source}:{line_no}: topics must be a unique list")
    if any(topic not in ALLOWED_TOPICS for topic in topics):
        fail(f"{source}:{line_no}: invalid topic value in {topics}")

    if labels["intent"] not in ALLOWED_INTENTS:
        fail(f"{source}:{line_no}: invalid intent {labels['intent']!r}")
    if labels["urgency"] not in ALLOWED_URGENCY:
        fail(f"{source}:{line_no}: invalid urgency {labels['urgency']!r}")

    if meta["source_type"] != "synthetic":
        fail(f"{source}:{line_no}: source_type must be 'synthetic'")
    if meta["difficulty"] not in ALLOWED_DIFFICULTY:
        fail(f"{source}:{line_no}: invalid difficulty {meta['difficulty']!r}")
    if not isinstance(meta["notes"], str) or not meta["notes"].strip():
        fail(f"{source}:{line_no}: meta.notes must be a non-empty string")

    if labels["is_ag_related"]:
        if not crops:
            fail(f"{source}:{line_no}: ag-related rows must include a crop label")
    else:
        if crops != []:
            fail(f"{source}:{line_no}: non-ag rows must use empty crops")
        if topics != []:
            fail(f"{source}:{line_no}: non-ag rows must use empty topics")
        if any(labels[key] for key in ["needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]):
            fail(f"{source}:{line_no}: non-ag rows cannot enable routing flags")


def main() -> None:
    if not SCHEMA_PATH.exists():
        fail(f"missing schema file: {SCHEMA_PATH}")

    seen_ids: set[str] = set()
    total_rows = 0
    for split_path in SPLITS:
        if not split_path.exists():
            fail(f"missing dataset split: {split_path}")
        with split_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    fail(f"{split_path}:{line_no}: blank lines are not allowed")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"{split_path}:{line_no}: invalid JSON: {exc}")
                validate_row(row, split_path, line_no, seen_ids)
                total_rows += 1

    print(f"Validated {total_rows} rows across {len(SPLITS)} splits with {len(seen_ids)} unique ids.")


if __name__ == "__main__":
    main()
