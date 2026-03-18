#!/usr/bin/env python3
"""Train a lightweight baseline routing model on sentence-transformer embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from routing_eval_lib import MODELS_DIR, evaluate_predictions, load_split, metrics_to_markdown, save_metrics, train_baseline_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split", default="train", help="Training split name without extension.")
    parser.add_argument("--val-split", default="val", help="Validation split name without extension.")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR, help="Directory for model artifacts.")
    args = parser.parse_args()

    train_bundle = load_split(args.train_split)
    val_bundle = load_split(args.val_split)
    model = train_baseline_model(train_bundle, val_bundle)

    val_predictions = model.predict(val_bundle.texts)
    val_metrics = evaluate_predictions(val_bundle, val_predictions)
    save_metrics(val_metrics, args.output_dir / "validation_metrics.json")
    (args.output_dir / "validation_metrics.md").write_text(metrics_to_markdown(val_metrics) + "\n", encoding="utf-8")

    summary = {
        "train_examples": len(train_bundle.rows),
        "validation_examples": len(val_bundle.rows),
        "artifacts": [
            "baseline_model.pkl",
            "metadata.json",
            "train_embeddings.npy",
            "val_embeddings.npy",
            "validation_metrics.json",
            "validation_metrics.md",
        ],
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved baseline model artifacts to {args.output_dir}")
    print(metrics_to_markdown(val_metrics))


if __name__ == "__main__":
    main()
