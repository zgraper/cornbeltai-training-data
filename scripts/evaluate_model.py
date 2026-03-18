#!/usr/bin/env python3
"""Evaluate the trained routing baseline on the test split and create confusion matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from routing_eval_lib import (
    ANALYSIS_DIR,
    CONFUSION_DIR,
    CROP_LABELS,
    INTENT_LABELS,
    MODELS_DIR,
    ROUTING_FLAGS,
    TOPIC_LABELS,
    BaselineModel,
    dataframe_from_predictions,
    evaluate_predictions,
    load_split,
    metrics_to_markdown,
    plot_confusion_matrix,
    save_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR, help="Directory containing baseline_model.pkl.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    args = parser.parse_args()

    model = BaselineModel.load(args.model_dir)
    bundle = load_split(args.split)
    predictions = model.predict(bundle.texts)
    metrics = evaluate_predictions(bundle, predictions)
    save_metrics(metrics, ANALYSIS_DIR / f"{args.split}_metrics.json")
    (ANALYSIS_DIR / f"{args.split}_metrics.md").write_text(metrics_to_markdown(metrics) + "\n", encoding="utf-8")

    frame = dataframe_from_predictions(bundle, predictions, args.split)
    frame.to_json(ANALYSIS_DIR / f"{args.split}_predictions.jsonl", orient="records", lines=True)
    frame.to_csv(ANALYSIS_DIR / f"{args.split}_predictions.csv", index=False)

    binary_labels = {
        "is_ag_related": ["0", "1"],
        **{flag: ["0", "1"] for flag in ROUTING_FLAGS},
    }
    for label, labels in binary_labels.items():
        plot_confusion_matrix(
            bundle.labels[label],
            predictions[label],
            labels,
            title=f"{label} confusion matrix ({args.split})",
            output_path=CONFUSION_DIR / f"{label}_{args.split}.png",
        )

    plot_confusion_matrix(
        bundle.labels["intent"],
        predictions["intent"],
        INTENT_LABELS,
        title=f"intent confusion matrix ({args.split})",
        output_path=CONFUSION_DIR / f"intent_{args.split}.png",
    )

    summary = {
        "split": args.split,
        "metrics_path": str((ANALYSIS_DIR / f"{args.split}_metrics.json").relative_to(args.model_dir.parents[1])),
        "prediction_rows": len(frame),
        "confusion_matrices": sorted(path.name for path in CONFUSION_DIR.glob(f"*_{args.split}.png")),
        "multilabel_targets": {"crops": CROP_LABELS, "topics": TOPIC_LABELS},
    }
    (ANALYSIS_DIR / f"{args.split}_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics_to_markdown(metrics))
    print(f"Saved confusion matrices to {CONFUSION_DIR}")


if __name__ == "__main__":
    main()
