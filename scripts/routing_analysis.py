#!/usr/bin/env python3
"""Analyze routing-flag correctness at the system level."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from routing_eval_lib import ANALYSIS_DIR, MODELS_DIR, BaselineModel, dataframe_from_predictions, load_split, routing_decision_summary


def markdown_breakdown(name: str, values: dict[str, dict[str, float]], top_n: int = 8) -> list[str]:
    ordered = sorted(values.items(), key=lambda item: (item[1]["correct_rate"], item[1]["total"]))
    lines = [f"### {name.title()} breakdown", "", "| Group | Total | Correct % | Over % | Under % | Mixed % |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, counts in ordered[:top_n]:
        lines.append(
            f"| {key} | {counts['total']} | {counts['correct_rate']:.1%} | {counts['over_rate']:.1%} | {counts['under_rate']:.1%} | {counts['mixed_rate']:.1%} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    bundle = load_split(args.split)
    model = BaselineModel.load(args.model_dir)
    predictions = model.predict(bundle.texts)
    frame = dataframe_from_predictions(bundle, predictions, args.split)
    summary = routing_decision_summary(frame)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / f"routing_summary_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Routing Analysis ({args.split})",
        "",
        f"- Exact routing match: {summary['exact_match_rate']:.1%}",
        f"- Over-triggering: {summary['over_trigger_rate']:.1%}",
        f"- Under-triggering: {summary['under_trigger_rate']:.1%}",
        f"- Mixed over/under errors: {summary['mixed_error_rate']:.1%}",
        "",
    ]
    for group_name in ["topic", "crop", "intent"]:
        lines.extend(markdown_breakdown(group_name, summary["breakdown"][group_name]))
    (ANALYSIS_DIR / f"routing_summary_{args.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
