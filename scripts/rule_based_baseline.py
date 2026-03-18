#!/usr/bin/env python3
"""Compare a keyword-based routing baseline against the trained model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from routing_eval_lib import (
    ANALYSIS_DIR,
    MODELS_DIR,
    BaselineModel,
    compare_metric_tables,
    evaluate_predictions,
    heuristic_predict,
    load_split,
    metrics_to_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    bundle = load_split(args.split)
    model = BaselineModel.load(args.model_dir)
    model_predictions = model.predict(bundle.texts)
    rule_predictions = heuristic_predict(bundle.rows)

    model_metrics = evaluate_predictions(bundle, model_predictions)
    rule_metrics = evaluate_predictions(bundle, rule_predictions)
    comparison = compare_metric_tables(model_metrics, rule_metrics)

    output = {
        "split": args.split,
        "model_metrics": model_metrics,
        "rule_metrics": rule_metrics,
        "comparison": comparison,
    }
    (ANALYSIS_DIR / f"rule_based_comparison_{args.split}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [f"# Rule-based vs model ({args.split})", "", "## Model metrics", "", metrics_to_markdown(model_metrics), "", "## Rule-based metrics", "", metrics_to_markdown(rule_metrics), "", "## Winner by label", "", "| Label | Metric | Model | Rules | Winner | Delta |", "| --- | --- | ---: | ---: | --- | ---: |"]
    for item in comparison:
        lines.append(
            f"| {item['label']} | {item['metric']} | {item['model']:.3f} | {item['rules']:.3f} | {item['winner']} | {item['delta']:.3f} |"
        )
    (ANALYSIS_DIR / f"rule_based_comparison_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
