# Routing Dataset

This directory contains the CornbeltAI routing/orchestration dataset used to train the lightweight entry-point classifier.

## Purpose

The dataset trains a lightweight classifier that runs at the start of the pipeline and predicts:

- whether a query is agriculture-related
- crop scope (`corn`, `soybean`, `both`, `unknown`, or none)
- agronomic topic labels
- routing flags for RAG, web search, weather data, farm data, and EPA pesticide labels
- optional intent and urgency labels

## Files

- `train.jsonl`: 2,400 examples
- `val.jsonl`: 400 examples
- `test.jsonl`: 400 examples
- `README.md`: dataset overview and workflow notes

## Notes

- The dataset has been audited for class balance, edge routing combinations, adversarial coverage, and duplicate inputs.
- Rows intentionally mix farmer-style phrasing, clipped short queries, vague field observations, mixed-intent prompts, and non-ag noise.
- `scripts/audit_report.txt` captures the baseline audit findings and the post-improvement summary.
- Validation and reporting utilities live in `scripts/validate_dataset.py` and `scripts/dataset_report.py`.

## Recommended workflow

```bash
python scripts/validate_dataset.py
python scripts/dataset_report.py
```
