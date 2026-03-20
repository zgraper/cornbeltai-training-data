# Routing Dataset Labeling Rules

This document defines the labeling policy for the first CornbeltAI routing dataset.
The goal is to help a lightweight model decide which downstream resources are needed,
not to answer the agronomic question itself.

## Example schema

Each row must be valid JSON and follow the schema in `schema/routing_schema.json`.

```json
{
  "id": "route_000001",
  "input": "Corn leaves are turning yellow after heavy rain. What should I do?",
  "labels": {
    "is_ag_related": true,
    "crops": ["corn"],
    "topics": ["nutrient", "weather"],
    "needs_rag": true,
    "needs_web_search": false,
    "needs_weather_data": true,
    "needs_farm_data": true,
    "needs_epa_label": false,
    "intent": "diagnosis",
    "urgency": "medium"
  },
  "meta": {
    "source_type": "synthetic",
    "difficulty": "medium",
    "notes": "yellowing after heavy rain may involve nitrogen loss and weather stress"
  }
}
```

## Core principles

- Label conservatively.
- Use only topics directly supported by the user query.
- Do not assume every agriculture-related question needs RAG.
- Distinguish time-sensitive lookups from general agronomic reasoning.
- Use `unknown` crop when agriculture is clear but crop context is missing.
- Use `[]` for `crops` and `topics` when a query is not agriculture-related.

## Field-by-field guidance

### `is_ag_related`
Set to `true` when the query is about farming, agronomy, crop production, farm operations,
ag economics, agricultural equipment, ag technology, or ag regulation.

Set to `false` for general consumer, office, travel, entertainment, coding, household,
or unrelated troubleshooting requests.

### `crops`
Allowed values:

- `corn`
- `soybean`
- `both`
- `unknown`

Rules:

- Use `["corn"]` or `["soybean"]` when a single crop is explicit.
- Use `["both"]` only when the query clearly applies to both corn and soybean.
- Use `["unknown"]` when the question is agricultural but the crop is not clear.
- Use `[]` when the query is not agriculture-related.
- Do not mix `both` with other crop labels.
- Do not mix `unknown` with explicit crop labels.

### `topics`
Choose zero or more labels from:

- `weather`
- `disease`
- `pest`
- `weed`
- `nutrient`
- `soil`
- `management`
- `market_economics`
- `equipment`
- `ag_technology`
- `policy_regulation`
- `general_agronomy`

Guidance:

- Use the smallest set of topics that captures the request.
- Do not add `general_agronomy` when a more specific topic already fully describes the query.
- Equipment-only questions may use only `equipment`.
- Current cash price or basis questions should use `market_economics`.
- Policy deadline or label compliance questions should use `policy_regulation`.
- Queries about monitors, sensors, prescriptions, drones, telematics, or data sync may use `ag_technology`.

### Routing flags

#### `needs_rag`
Set to `true` when the query likely needs agronomic knowledge retrieval, including:

- disease, pest, weed, or nutrient interpretation
- symptom diagnosis
- crop management decisions
- stage-specific recommendations
- soil or agronomic best-practice guidance

Set to `false` when the query is:

- non-agricultural
- a simple current lookup
- a straightforward operational or equipment question that does not require agronomic retrieval

#### `needs_web_search`
Set to `true` when the query depends on current or live information, for example:

- today’s cash price, basis, or futures context
- current policy or regulation updates
- latest pest or disease outbreak reports
- recent local elevator or market status
- current news or supply chain disruptions

Set to `false` when the question can be routed without live information.

#### `needs_weather_data`
Set to `true` when the query depends on weather or forecast conditions, including:

- rain timing
- frost, heat, drought, humidity, wind
- field workability
- spray windows
- planting or harvest timing
- drying conditions
- GDD-like or weather-driven crop stress questions

#### `needs_farm_data`
Set to `true` when field-specific context would materially change the answer, such as:

- planting date
- growth stage
- hybrid or variety
- soil tests
- field history
- location
- prior applications
- yield history
- local observations from the field

#### `needs_epa_label`
Set to `true` when the system likely needs an EPA pesticide label or label-derived directions, including:

- herbicide, insecticide, fungicide, or pesticide product questions
- Section 24(c), dicamba cutoff, drift, REI, PHI, adjuvant, tank-mix, or rate restrictions
- spray decisions where pesticide label compliance materially affects the answer

Set to `false` when the query does not involve pesticide label use or can be routed without label-specific guidance.

### `intent`
Use exactly one label:

- `question`: broad factual question without strong lookup/diagnosis planning behavior
- `diagnosis`: symptom or problem identification
- `recommendation`: asks what action should be taken
- `planning`: forward-looking timing or operational planning
- `monitoring`: asks what to watch, track, scout, or check
- `information_lookup`: requests a lookup of current or reference information
- `comparison`: compares products, practices, prices, hybrids, or strategies
- `other`: everything else

### `urgency`
Use:

- `high` when the user implies immediate action, active damage, or a same-day time decision
- `medium` when the issue matters soon but is not clearly immediate
- `low` when the request is informational, strategic, or non-urgent

## Common edge cases

### Ag-related but crop unknown
Example: `Should I wait for the field to dry before side-dressing?`

- `is_ag_related = true`
- `crops = ["unknown"]`

### Weather-only routing without RAG
Example: `Will I have a spray window tomorrow afternoon?`

- Often `needs_weather_data = true`
- May keep `needs_rag = false` if the request is just a timing lookup and not an agronomic interpretation

### Market lookup
Example: `What are central Illinois soybean cash bids today?`

- `needs_web_search = true`
- Usually `needs_rag = false`
- `topics = ["market_economics"]`

### Equipment question that is ag-related
Example: `My planter monitor is not reading population on one row.`

- `is_ag_related = true`
- `topics` may be `["equipment", "ag_technology"]`
- `needs_rag` may remain `false`

### Non-ag noise
Example: `Can you help me reset my email password?`

- `is_ag_related = false`
- `crops = []`
- `topics = []`
- all routing flags `false`, including `needs_epa_label`

## Quality checklist

Before finalizing a dataset split:

1. Validate every line against the JSON schema.
2. Check that IDs are unique across all splits.
3. Confirm `crops = []` and `topics = []` for non-ag rows.
4. Confirm label combinations match the wording of the query.
5. Avoid clusters of near-duplicate wording.
6. Keep the overall ag/non-ag balance within the requested range.
