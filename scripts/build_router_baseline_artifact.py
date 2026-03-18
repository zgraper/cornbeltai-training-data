#!/usr/bin/env python3
"""Train a lightweight token-based routing artifact for local inference without external ML dependencies."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

from src.router import (
    BINARY_LABELS,
    CROP_LABELS,
    DEFAULT_MODEL_DIR,
    INTENT_LABELS,
    MODEL_FILENAME,
    MULTICLASS_FIELDS,
    MULTILABEL_FIELDS,
    ROUTING_FLAGS,
    TOPIC_LABELS,
    URGENCY_LABELS,
    preprocess_text,
)

DATASET_DIR = Path("datasets/routing")
TRAINING_SPLITS = ["train", "val"]



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows



def load_rows(splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        rows.extend(read_jsonl(DATASET_DIR / f"{split}.jsonl"))
    return rows



def label_vector(row: dict[str, Any], field: str) -> list[str]:
    labels = row["labels"]
    if field in {"crops", "topics"}:
        return list(labels[field])
    return [str(labels[field])]



def build_one_vs_rest(rows: list[dict[str, Any]], field: str, classes: list[str]) -> dict[str, Any]:
    total_docs = len(rows)
    total_feature_counts: Counter[str] = Counter()
    class_doc_counts: Counter[str] = Counter()
    class_feature_counts: dict[str, Counter[str]] = {label: Counter() for label in classes}

    for row in rows:
        features = preprocess_text(row["input"])
        feature_set = set(features)
        total_feature_counts.update(feature_set)
        active_labels = set(label_vector(row, field))
        for label in classes:
            if label in active_labels:
                class_doc_counts[label] += 1
                class_feature_counts[label].update(feature_set)

    models: dict[str, Any] = {}
    min_weight = 0.15
    max_features = 120
    for label in classes:
        pos_docs = class_doc_counts[label]
        neg_docs = max(total_docs - pos_docs, 0)
        bias = math.log((pos_docs + 1) / (neg_docs + 1))
        weights: dict[str, float] = {}
        scored_features = []
        for feature, global_count in total_feature_counts.items():
            pos_count = class_feature_counts[label][feature]
            neg_count = global_count - pos_count
            weight = math.log((pos_count + 1) / (neg_count + 1))
            if abs(weight) >= min_weight:
                scored_features.append((abs(weight), feature, round(weight, 4)))
        scored_features.sort(reverse=True)
        for _, feature, weight in scored_features[:max_features]:
            weights[feature] = weight
        models[label] = {"bias": round(bias, 4), "weights": weights}
    return models



def build_binary_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    binary_models: dict[str, Any] = {}
    for field in BINARY_LABELS:
        adapted_rows = []
        for row in rows:
            cloned = {"input": row["input"], "labels": {field: "true" if row["labels"][field] else "false"}}
            adapted_rows.append(cloned)
        binary_models[field] = build_one_vs_rest(adapted_rows, field, ["true", "false"])["true"]
    return binary_models



def build_thresholds() -> dict[str, Any]:
    return {
        "binary": {
            "is_ag_related": 0.5,
            "needs_rag": 0.52,
            "needs_web_search": 0.5,
            "needs_weather_data": 0.5,
            "needs_farm_data": 0.5,
        },
        "crops": 0.45,
        "topics": 0.42,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    rows = load_rows(TRAINING_SPLITS)
    artifact = {
        "model_type": "lightweight_token_log_odds",
        "training_splits": TRAINING_SPLITS,
        "training_examples": len(rows),
        "preprocessing": {
            "normalization": "lowercase + whitespace collapse",
            "features": ["unigram", "bigram"],
        },
        "thresholds": build_thresholds(),
        "binary_models": build_binary_models(rows),
        "multilabel_models": {
            "crops": build_one_vs_rest(rows, "crops", CROP_LABELS),
            "topics": build_one_vs_rest(rows, "topics", TOPIC_LABELS),
        },
        "multiclass_models": {
            "intent": build_one_vs_rest(rows, "intent", INTENT_LABELS),
            "urgency": build_one_vs_rest(rows, "urgency", URGENCY_LABELS),
        },
        "label_spaces": {
            "routing_flags": ROUTING_FLAGS,
            "crops": CROP_LABELS,
            "topics": TOPIC_LABELS,
            "intent": INTENT_LABELS,
            "urgency": URGENCY_LABELS,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / MODEL_FILENAME).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": artifact["model_type"],
                "training_splits": TRAINING_SPLITS,
                "training_examples": len(rows),
                "preprocessing": artifact["preprocessing"],
                "thresholds": artifact["thresholds"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote lightweight routing artifact to {args.output_dir / MODEL_FILENAME}")


if __name__ == "__main__":
    main()
