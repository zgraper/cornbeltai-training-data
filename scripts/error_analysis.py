#!/usr/bin/env python3
"""Summarize and sample the model's failure cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from routing_eval_lib import ERROR_DIR, MODELS_DIR, ROUTING_FLAGS, BaselineModel, dataframe_from_predictions, load_split


def collect_error_records(frame: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for _, row in frame.iterrows():
        for label in ["is_ag_related", *ROUTING_FLAGS, "intent", "urgency"]:
            true_value = row[f"true_{label}"]
            pred_value = row[f"pred_{label}"]
            if true_value != pred_value:
                error_type = "false_positive" if str(true_value) in {"0", "False"} and str(pred_value) in {"1", "True"} else "false_negative"
                if label in {"intent", "urgency"}:
                    error_type = "mismatch"
                records.append(
                    {
                        "id": row["id"],
                        "input": row["input"],
                        "label": label,
                        "error_type": error_type,
                        "true_value": true_value,
                        "pred_value": pred_value,
                        "topic": ", ".join(row["true_topics"]) if row["true_topics"] else "none",
                        "crop": ", ".join(row["true_crops"]) if row["true_crops"] else "none",
                        "intent": row["true_intent"],
                    }
                )
        if row["true_crops"] != row["pred_crops"]:
            records.append(
                {
                    "id": row["id"],
                    "input": row["input"],
                    "label": "crops",
                    "error_type": "set_mismatch",
                    "true_value": row["true_crops"],
                    "pred_value": row["pred_crops"],
                    "topic": ", ".join(row["true_topics"]) if row["true_topics"] else "none",
                    "crop": ", ".join(row["true_crops"]) if row["true_crops"] else "none",
                    "intent": row["true_intent"],
                }
            )
        if row["true_topics"] != row["pred_topics"]:
            records.append(
                {
                    "id": row["id"],
                    "input": row["input"],
                    "label": "topics",
                    "error_type": "set_mismatch",
                    "true_value": row["true_topics"],
                    "pred_value": row["pred_topics"],
                    "topic": ", ".join(row["true_topics"]) if row["true_topics"] else "none",
                    "crop": ", ".join(row["true_crops"]) if row["true_crops"] else "none",
                    "intent": row["true_intent"],
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    bundle = load_split(args.split)
    model = BaselineModel.load(args.model_dir)
    predictions = model.predict(bundle.texts)
    frame = dataframe_from_predictions(bundle, predictions, args.split)
    errors = pd.DataFrame.from_records(collect_error_records(frame))

    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(ERROR_DIR / f"errors_{args.split}.csv", index=False)
    grouped = {
        "by_label": errors.groupby("label").size().sort_values(ascending=False).to_dict(),
        "by_topic": errors.groupby("topic").size().sort_values(ascending=False).head(20).to_dict(),
        "by_crop": errors.groupby("crop").size().sort_values(ascending=False).to_dict(),
        "by_intent": errors.groupby("intent").size().sort_values(ascending=False).to_dict(),
        "top_50_failure_cases": errors.head(50).to_dict(orient="records"),
        "false_positive_examples": errors[errors["error_type"] == "false_positive"].head(10).to_dict(orient="records"),
        "false_negative_examples": errors[errors["error_type"] == "false_negative"].head(10).to_dict(orient="records"),
    }
    (ERROR_DIR / f"errors_{args.split}.json").write_text(json.dumps(grouped, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Error Analysis ({args.split})",
        "",
        "## Error counts by label",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label, count in grouped["by_label"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Top 50 failure cases", ""])
    for record in grouped["top_50_failure_cases"]:
        lines.append(
            f"- `{record['id']}` [{record['label']}/{record['error_type']}] true={record['true_value']} pred={record['pred_value']} :: {record['input']}"
        )
    lines.extend(["", "## Pattern hotspots", ""])
    for title, key in [("Topics", "by_topic"), ("Crops", "by_crop"), ("Intent", "by_intent")]:
        lines.append(f"### {title}")
        lines.append("")
        for name, count in grouped[key].items():
            lines.append(f"- {name}: {count}")
        lines.append("")
    (ERROR_DIR / f"errors_{args.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
