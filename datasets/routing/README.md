# Routing Dataset

This directory contains the first CornbeltAI routing/orchestration dataset.

## Purpose

The dataset trains a lightweight classifier that runs at the start of the pipeline and predicts:

- whether a query is agriculture-related
- crop scope (`corn`, `soybean`, `both`, `unknown`, or none)
- agronomic topic labels
- routing flags for RAG, web search, weather data, and farm data
- optional intent and urgency labels

## Files

- `train.jsonl`: 900 examples
- `val.jsonl`: 150 examples
- `test.jsonl`: 150 examples

## Notes

- All rows are synthetic and schema-validated.
- The dataset mixes farmer-style phrasing, formal agronomy wording, ambiguous wording, and non-ag noise.
- Validation and reporting utilities live in `scripts/validate_dataset.py` and `scripts/dataset_report.py`.

## Recommended workflow

```bash
python scripts/validate_dataset.py
python scripts/dataset_report.py
```
