# CornbeltAI Routing Evaluation Summary

## Dataset overview

- Total examples: 1,600
- Splits: train 1,200 / validation 200 / test 200
- Ag-related vs non-ag: 1,120 ag-related (70.0%) / 480 non-ag (30.0%)
- Positive routing labels:
  - `needs_rag`: 597
  - `needs_web_search`: 240
  - `needs_weather_data`: 338
  - `needs_farm_data`: 643
- Crop balance:
  - `corn`: 366
  - `soybean`: 255
  - `both`: 170
  - `unknown`: 329
- Most common topics:
  - `weather`: 348
  - `management`: 193
  - `general_agronomy`: 146
  - `market_economics`: 135
  - `nutrient`: 130

## Model performance

This repository now includes a full baseline training and evaluation harness:

- `scripts/train_baseline_model.py`
- `scripts/evaluate_model.py`
- `scripts/routing_analysis.py`
- `scripts/rule_based_baseline.py`
- `scripts/error_analysis.py`

The intended workflow is:

```bash
python scripts/train_baseline_model.py
python scripts/evaluate_model.py
python scripts/routing_analysis.py
python scripts/rule_based_baseline.py
python scripts/error_analysis.py
```

Generated artifacts are designed to land in:

- `models/baseline/`
- `reports/confusion_matrices/`
- `reports/analysis/`
- `reports/error_analysis/`

### Current environment note

This execution environment did not have the required Python ML packages installed, and package installation failed because outbound package index access was blocked. Because of that, trained-model metrics and confusion matrix images could not be generated in-session.

Once dependencies are available, `scripts/evaluate_model.py` will produce per-label accuracy / precision / recall / F1, plus micro/macro F1 for `crops` and `topics`.

## Routing performance

`scripts/routing_analysis.py` evaluates system-level routing correctness by comparing the four routing flags as a bundle and reports:

- exact routing decision rate
- over-triggering rate
- under-triggering rate
- mixed error rate
- breakdowns by topic, crop, and intent

## Comparison: model vs rule-based baseline

`scripts/rule_based_baseline.py` compares the trained embedding + logistic-regression baseline against a simple keyword heuristic system and reports:

- per-label winner
- labels where rules outperform the model
- labels where the model outperforms rules
- metric deltas for each target

## Error analysis

`scripts/error_analysis.py` groups failures by:

- label type
- topic
- crop
- intent

It also exports:

- top 50 failure cases
- false positive examples
- false negative examples
- CSV + JSON + Markdown summaries

## Recommendations

Based on the dataset distribution and the new harness design:

1. Add more low-frequency examples for `policy_regulation`, `equipment`, and `ag_technology` to reduce label sparsity.
2. Watch the `unknown` crop label closely; it is common enough that confusion with explicit crop labels could mask routing weaknesses.
3. Compare `planning`, `recommendation`, and `diagnosis` errors first, since these intents are likely to drive downstream orchestration differently.
4. Inspect `needs_farm_data` and `needs_rag` together, because they are both common and likely to be over-triggered by symptom-heavy prompts.
5. Use the rule-based comparison to identify label definitions that may still be too ambiguous for a lightweight first-pass model.
