#!/usr/bin/env python3
"""Validate the CornbeltAI routing dataset with schema, semantics, duplicates, and distribution checks."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "routing"
SCHEMA_PATH = ROOT / "schema" / "routing_schema.json"
SPLITS = [DATASET_DIR / "train.jsonl", DATASET_DIR / "val.jsonl", DATASET_DIR / "test.jsonl"]
EXPECTED_SPLIT_SIZES = {"train.jsonl": 2460, "val.jsonl": 410, "test.jsonl": 410}

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


class ValidationError(Exception):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9? ]", "", lowered)
    return lowered


def validate_against_schema(value: object, schema: dict, context: str) -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            fail(f"{context}: expected object")
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                fail(f"{context}: missing required key {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra_keys = set(value) - set(properties)
            if extra_keys:
                fail(f"{context}: unexpected keys {sorted(extra_keys)}")
        for key, subschema in properties.items():
            if key in value:
                validate_against_schema(value[key], subschema, f"{context}.{key}")
        return

    if schema_type == "array":
        if not isinstance(value, list):
            fail(f"{context}: expected array")
        if schema.get("uniqueItems") and len(value) != len(set(json.dumps(item, sort_keys=True) for item in value)):
            fail(f"{context}: expected unique array items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_against_schema(item, item_schema, f"{context}[{index}]")
        return

    if schema_type == "string":
        if not isinstance(value, str):
            fail(f"{context}: expected string")
        if len(value) < schema.get("minLength", 0):
            fail(f"{context}: string shorter than minLength")
        pattern = schema.get("pattern")
        if pattern and not re.fullmatch(pattern, value):
            fail(f"{context}: value {value!r} does not match pattern {pattern!r}")
        enum = schema.get("enum")
        if enum and value not in enum:
            fail(f"{context}: value {value!r} not in enum {enum}")
        return

    if schema_type == "boolean":
        if not isinstance(value, bool):
            fail(f"{context}: expected boolean")
        return

    fail(f"{context}: unsupported schema type {schema_type!r}")


def validate_row(row: dict, source: Path, line_no: int, seen_ids: set[str], input_index: dict[str, str], route_signatures: set[str]) -> None:
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

    input_text = row["input"]
    if not isinstance(input_text, str) or not input_text.strip():
        fail(f"{source}:{line_no}: input must be a non-empty string")
    normalized_input = normalize_text(input_text)
    if normalized_input in input_index:
        fail(f"{source}:{line_no}: duplicate normalized input also seen in {input_index[normalized_input]}")
    input_index[normalized_input] = f"{source}:{line_no}"

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
        if not topics:
            fail(f"{source}:{line_no}: ag-related rows must include at least one topic")
    else:
        if crops != []:
            fail(f"{source}:{line_no}: non-ag rows must use empty crops")
        if topics != []:
            fail(f"{source}:{line_no}: non-ag rows must use empty topics")
        if any(labels[key] for key in ["needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]):
            fail(f"{source}:{line_no}: non-ag rows cannot enable routing flags")

    signature = json.dumps(
        {
            "input": normalized_input,
            "labels": labels,
            "meta_notes": meta["notes"],
        },
        sort_keys=True,
    )
    if signature in route_signatures:
        fail(f"{source}:{line_no}: duplicate input/label signature detected")
    route_signatures.add(signature)


def compute_stats(rows: list[dict]) -> dict:
    ag_rows = [row for row in rows if row["labels"]["is_ag_related"]]
    total = len(rows)
    route_combo_counts = Counter()
    topic_counts = Counter()
    short_queries = 0
    vague_queries = 0
    multi_topic = 0
    hard_examples = 0
    for row in rows:
        labels = row["labels"]
        combo = (labels["needs_rag"], labels["needs_web_search"], labels["needs_weather_data"], labels["needs_farm_data"])
        route_combo_counts[combo] += 1
        for topic in labels["topics"]:
            topic_counts[topic] += 1
        if len(row["input"].split()) <= 5:
            short_queries += 1
        if VAGUE_REGEX.search(row["input"].lower()):
            vague_queries += 1
        if len(labels["topics"]) > 1:
            multi_topic += 1
        if row["meta"]["difficulty"] == "hard":
            hard_examples += 1

    return {
        "total": total,
        "ag_total": len(ag_rows),
        "non_ag_total": total - len(ag_rows),
        "route_combo_counts": route_combo_counts,
        "topic_counts": topic_counts,
        "short_queries": short_queries,
        "vague_queries": vague_queries,
        "multi_topic": multi_topic,
        "hard_examples": hard_examples,
        "rag_ag_ratio": sum(1 for row in ag_rows if row["labels"]["needs_rag"]) / max(len(ag_rows), 1),
    }


def run_distribution_checks(stats: dict) -> list[str]:
    warnings: list[str] = []
    total = stats["total"]
    non_ag_ratio = stats["non_ag_total"] / total
    if not 0.25 <= non_ag_ratio <= 0.35:
        fail(f"distribution: non-ag ratio {non_ag_ratio:.1%} is outside the expected 25-35% band")

    if not 0.30 <= stats["rag_ag_ratio"] <= 0.70:
        fail(f"distribution: needs_rag ratio among ag rows {stats['rag_ag_ratio']:.1%} is outside the expected 30-70% band")

    short_ratio = stats["short_queries"] / total
    if short_ratio < 0.08:
        fail(f"distribution: short query ratio {short_ratio:.1%} is below the 8% minimum")

    vague_ratio = stats["vague_queries"] / total
    if vague_ratio < 0.04:
        fail(f"distribution: vague query ratio {vague_ratio:.1%} is below the 4% minimum")

    multi_ratio = stats["multi_topic"] / total
    if multi_ratio < 0.20:
        fail(f"distribution: multi-topic ratio {multi_ratio:.1%} is below the 20% minimum")

    if stats["hard_examples"] < 100:
        fail(f"distribution: hard example count {stats['hard_examples']} is below the required minimum of 100")

    for combo_name, combo in EDGE_COMBOS.items():
        count = stats["route_combo_counts"][combo]
        if count < 20:
            fail(f"distribution: edge routing combo {combo_name} has only {count} rows; expected at least 20")

    low_topic_threshold = max(20, int(stats["ag_total"] * 0.04))
    for topic in sorted(ALLOWED_TOPICS):
        count = stats["topic_counts"][topic]
        if count < low_topic_threshold:
            warnings.append(f"warning: topic {topic} has only {count} rows (< {low_topic_threshold})")

    return warnings


def main() -> None:
    if not SCHEMA_PATH.exists():
        fail(f"missing schema file: {SCHEMA_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    seen_ids: set[str] = set()
    input_index: dict[str, str] = {}
    route_signatures: set[str] = set()
    split_counts = Counter()
    rows: list[dict] = []

    try:
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
                    validate_against_schema(row, schema, f"{split_path.name}:{line_no}")
                    validate_row(row, split_path, line_no, seen_ids, input_index, route_signatures)
                    split_counts[split_path.name] += 1
                    rows.append(row)
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    for split_name, expected_count in EXPECTED_SPLIT_SIZES.items():
        actual_count = split_counts[split_name]
        if actual_count != expected_count:
            print(f"ERROR: split {split_name} has {actual_count} rows; expected {expected_count}")
            raise SystemExit(1)

    stats = compute_stats(rows)
    try:
        warnings = run_distribution_checks(stats)
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    print(f"Validated {len(rows)} rows across {len(SPLITS)} splits with {len(seen_ids)} unique ids.")
    print("Distribution checks passed.")
    for warning in warnings:
        print(warning)


if __name__ == "__main__":
    main()
