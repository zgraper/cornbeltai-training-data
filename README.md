# CornbeltAI Training Data

This repository contains training datasets for lightweight models used inside the CornbeltAI platform.

## Purpose

The first dataset in this repository is a **routing/orchestration dataset**. Its purpose is to support a small, fast model that runs at the **start of the CornbeltAI pipeline**.

This model does **not** answer the user’s agronomic question directly.

Instead, it determines how the full system should respond.

## Why this dataset matters

CornbeltAI will eventually support multiple downstream resources and tools, including:

- internal agronomy RAG retrieval
- web retrieval for current or live information
- weather-based reasoning
- farm-specific contextual reasoning

These resources are useful, but they also add latency, cost, and complexity.

The routing model exists to make an early decision about which resources are actually needed for a given user query.

This helps the platform:

- reduce unnecessary LLM calls
- reduce unnecessary retrieval steps
- improve speed
- improve consistency
- support cheaper first-pass inference

## Primary task

The dataset should help a small model classify a user query into a structured routing decision.

Each example should indicate:

- whether the query is agriculture-related
- which crop or crops the query refers to
- which agronomic topics are involved
- whether the system should access:
  - RAG retrieval
  - live or current web information
  - weather data
  - farm-specific data
  - EPA pesticide label retrieval

Optional labels such as intent and urgency may also be included if they improve downstream orchestration.

## Initial label design

### Core labels
- `is_ag_related`
- `crops`
- `topics`
- `needs_rag`
- `needs_web_search`
- `needs_weather_data`
- `needs_farm_data`
- `needs_epa_label`

### Optional supporting labels
- `intent`
- `urgency`

## Crop scope

The initial routing dataset is focused on:

- corn
- soybean

Queries may also be:
- relevant to both
- agriculturally relevant but crop-unknown
- not agriculture-related at all

## Topic scope

The initial topic taxonomy may include:

- weather
- disease
- pest
- weed
- nutrient
- soil
- management
- market_economics
- equipment
- ag_technology
- policy_regulation
- general_agronomy

This taxonomy may evolve over time.

## Routing philosophy

The model should act as a **control layer**, not a knowledge layer.

That means the model should answer questions like:

- Is this even agriculture-related?
- Does this query require agronomic retrieval?
- Is this about current conditions that require the web?
- Does this depend on weather?
- Does this require farm-specific context?
- Does this require retrieving an EPA pesticide label?
- What crop and topic categories are implicated?

The model should **not** attempt to generate agronomic advice.

## Labeling philosophy

Labels should be assigned conservatively and consistently.

### `needs_rag`
Use `true` when the query likely requires agronomic knowledge retrieval, such as:
- diagnosis
- management recommendations
- disease/pest/weed/nutrient interpretation
- agronomic best practices

### `needs_web_search`
Use `true` when the query depends on current or live information, such as:
- today’s prices
- recent outbreaks
- current regulation updates
- latest news
- current local conditions not already stored internally

### `needs_weather_data`
Use `true` when the question depends on forecast or weather conditions, such as:
- planting timing
- spray timing
- rain, drought, heat, frost, humidity, wind
- field workability
- weather-driven crop stress

### `needs_farm_data`
Use `true` when the question depends on field-specific context, such as:
- planting date
- field history
- soil tests
- location
- growth stage
- yield history
- hybrid/variety
- user-entered farm records

### `needs_epa_label`
Use `true` when the query likely requires an EPA pesticide label or label-derived restrictions, such as:
- herbicide, insecticide, fungicide, or pesticide use questions
- label compliance, cutoff, drift, REI, PHI, or tank-mix restrictions
- spray decisions where legal label directions materially affect the answer

## Dataset design goals

The dataset should be:

- realistic
- diverse
- balanced
- useful for lightweight model training
- structured for automated validation
- versionable and reproducible

It should include:

- normal farmer-style wording
- vague user questions
- symptom-based queries
- planning questions
- market questions
- weather questions
- non-ag noise examples
- ambiguous examples
- edge cases involving multiple routing flags

## Recommended dataset structure

```json
{
  "id": "route_000001",
  "input": "Should I spray before the rain tomorrow?",
  "labels": {
    "is_ag_related": true,
    "crops": ["unknown"],
    "topics": ["management", "weather"],
    "needs_rag": true,
    "needs_web_search": false,
    "needs_weather_data": true,
    "needs_farm_data": true,
    "needs_epa_label": false,
    "intent": "planning",
    "urgency": "medium"
  },
  "meta": {
    "source_type": "synthetic",
    "difficulty": "medium",
    "notes": "spray timing depends on agronomic practice, weather, and likely farm context"
  }
}
```

## Repository layout

Recommended structure:
```
cornbeltai-training-data/
  datasets/
    routing/
      train.jsonl
      val.jsonl
      test.jsonl
      README.md
  schema/
    routing_schema.json
    labeling_rules.md
  scripts/
    validate_dataset.py
    dataset_report.py
```

## Non-goals for this repository stage

At this stage, this repository is not focused on:

- final answer generation
- large language model fine-tuning
- full agronomic recommendation generation
- long-form question answering
- structured for automated validation
- production inference APIs

The current focus is only on **training data for routing and orchestration.**

## Guidance for coding agents

When generating or modifying data in this repository:

1. Preserve schema consistency.
2. Prefer realistic phrasing over artificial phrasing.
3. Avoid repetitive near-duplicates.
4. Keep topic labels conservative.
5. Do not assume every ag-related question needs RAG.
6. Include enough non-ag examples to reduce false positives.
7. Validate outputs with scripts before finalizing.
8. Favor data quality over raw dataset size.

## Long-term roadmap

Later datasets in this repository may include:

- entity extraction
- severity classification
- actionability classification
- crop-stage extraction
- symptom-to-topic mapping
- retrieval evaluation sets
- synthetic and semi-synthetic agronomy benchmark tasks

The routing dataset is the first layer because it determines how the rest of the CornbeltAI system behaves.

