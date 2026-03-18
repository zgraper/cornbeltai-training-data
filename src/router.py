from __future__ import annotations

import json
import math
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
DEFAULT_LOG_PATH = LOGS_DIR / "routing_logs.jsonl"
DEFAULT_MODEL_DIR = ROOT / "models" / "baseline"
MODEL_FILENAME = "router_model.json"

ROUTING_FLAGS = ["needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]
BINARY_LABELS = ["is_ag_related", *ROUTING_FLAGS]
CROP_LABELS = ["corn", "soybean", "both", "unknown"]
TOPIC_LABELS = [
    "weather",
    "disease",
    "pest",
    "weed",
    "nutrient",
    "soil",
    "management",
    "market_economics",
    "equipment",
    "ag_technology",
    "policy_regulation",
    "general_agronomy",
]
INTENT_LABELS = [
    "question",
    "diagnosis",
    "recommendation",
    "planning",
    "monitoring",
    "information_lookup",
    "comparison",
    "other",
]
URGENCY_LABELS = ["low", "medium", "high"]
MULTILABEL_FIELDS = {"crops": CROP_LABELS, "topics": TOPIC_LABELS}
MULTICLASS_FIELDS = {"intent": INTENT_LABELS, "urgency": URGENCY_LABELS}
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

HEURISTIC_KEYWORDS = {
    "needs_web_search": [
        "today", "right now", "current", "latest", "cash bid", "cash bids", "basis", "futures", "price", "prices",
        "regulation", "cutoff", "deadline", "dicamba", "news", "outbreak",
    ],
    "needs_weather_data": [
        "rain", "storm", "temperature", "temperatures", "forecast", "humidity", "wind", "frost", "heat", "drought",
        "spray window", "spray today", "tomorrow", "field to dry", "dry spell", "weather", "plant this week", "this week", "should i plant", "plant?",
    ],
    "needs_rag": [
        "disease", "fungus", "fungal", "blight", "rot", "yellow", "deficiency", "armyworm", "aphid", "waterhemp",
        "marestail", "foxtail", "lambsquarters", "weed", "pest", "soil crusting", "side-dress", "sidedress",
        "what should i do", "next move", "spray", "plant", "planting",
    ],
    "needs_farm_data": [
        "field", "farm", "quarter", "north 80", "west farm", "soil test", "planting date", "growth stage", "hybrid",
        "variety", "yield history", "home quarter", "river bottom", "sandy hill", "black dirt", "should i spray", "sidedress",
    ],
}
TOPIC_KEYWORDS = {
    "weather": ["rain", "forecast", "wind", "humidity", "temperature", "frost", "heat", "drought", "weather", "tomorrow"],
    "disease": ["disease", "blight", "fungus", "rot", "mold", "lesion"],
    "pest": ["pest", "armyworm", "aphid", "beetle", "worm", "insect"],
    "weed": ["weed", "waterhemp", "foxtail", "marestail", "lambsquarters", "palmer"],
    "nutrient": ["nitrogen", "phosphorus", "potassium", "sulfur", "nutrient", "deficiency", "yellow", "sidedress", "side-dress"],
    "soil": ["soil", "infiltration", "compaction", "crusting", "ph", "cec"],
    "management": ["spray", "planting", "plant", "harvest", "side-dress", "sidedress", "manage", "rotation"],
    "market_economics": ["cash bid", "cash bids", "price", "basis", "futures", "market", "elevator"],
    "equipment": ["planter", "monitor", "row unit", "closing wheel", "combine", "toolbar"],
    "ag_technology": ["sensor", "telematics", "prescription", "monitor", "drone", "data sync", "app"],
    "policy_regulation": ["regulation", "cutoff", "label", "compliance", "deadline", "dicamba"],
    "general_agronomy": ["agronomy", "crop", "field"],
}
CROP_KEYWORDS = {
    "corn": ["corn", "maize"],
    "soybean": ["soybean", "soybeans", "bean", "beans"],
}
INTENT_KEYWORDS = {
    "diagnosis": ["what is", "what does that point to", "diagnose", "suspect", "why is"],
    "recommendation": ["what should i do", "do i need to", "next move", "can i still", "fix"],
    "planning": ["should i", "tomorrow", "before next season", "timing", "window", "this week"],
    "monitoring": ["watch", "track", "monitor", "scout", "doing"],
    "information_lookup": ["current", "today", "right now", "lookup", "what is the current", "price of"],
    "comparison": ["compare", "versus", "vs", "better"],
    "question": ["what", "why", "how"],
}
NON_AG_HINTS = ["movie", "restaurant", "song", "flight", "hotel", "traffic", "game", "weather in paris"]


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _heuristic_overrides(text: str) -> dict[str, Any]:
    lowered = text.lower()
    ag_keywords = [keyword for values in CROP_KEYWORDS.values() for keyword in values]
    ag_keywords += [keyword for values in TOPIC_KEYWORDS.values() for keyword in values]
    ag_keywords += [keyword for values in HEURISTIC_KEYWORDS.values() for keyword in values]
    heuristic_ag = _contains_any(lowered, ag_keywords)
    if _contains_any(lowered, NON_AG_HINTS):
        heuristic_ag = False

    overrides: dict[str, Any] = {
        "is_ag_related": heuristic_ag,
        "needs_rag": heuristic_ag and _contains_any(lowered, HEURISTIC_KEYWORDS["needs_rag"]),
        "needs_web_search": heuristic_ag and _contains_any(lowered, HEURISTIC_KEYWORDS["needs_web_search"]),
        "needs_weather_data": heuristic_ag and _contains_any(lowered, HEURISTIC_KEYWORDS["needs_weather_data"]),
        "needs_farm_data": heuristic_ag and _contains_any(lowered, HEURISTIC_KEYWORDS["needs_farm_data"]),
    }

    has_corn = _contains_any(lowered, CROP_KEYWORDS["corn"])
    has_soybean = _contains_any(lowered, CROP_KEYWORDS["soybean"])
    if has_corn and has_soybean:
        overrides["crops"] = ["both"]
    elif has_corn:
        overrides["crops"] = ["corn"]
    elif has_soybean:
        overrides["crops"] = ["soybean"]
    elif heuristic_ag:
        overrides["crops"] = ["unknown"]
    else:
        overrides["crops"] = []

    topic_matches = [topic for topic, keywords in TOPIC_KEYWORDS.items() if _contains_any(lowered, keywords)]
    overrides["topics"] = topic_matches or (["general_agronomy"] if heuristic_ag else [])

    intent = "other"
    for candidate, keywords in INTENT_KEYWORDS.items():
        if _contains_any(lowered, keywords):
            intent = candidate
            break
    if not heuristic_ag and intent == "other":
        intent = "question" if "?" in lowered else "other"
    overrides["intent"] = intent

    urgency = "low"
    if _contains_any(lowered, ["now", "today", "tomorrow", "urgent", "right now"]):
        urgency = "high"
    elif _contains_any(lowered, ["soon", "next", "this week"]):
        urgency = "medium"
    elif heuristic_ag:
        urgency = "medium"
    overrides["urgency"] = urgency
    return overrides



def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()



def preprocess_text(text: str) -> list[str]:
    normalized = " ".join(text.lower().strip().split())
    unigrams = TOKEN_PATTERN.findall(normalized)
    bigrams = [f"{unigrams[index]}_{unigrams[index + 1]}" for index in range(len(unigrams) - 1)]
    return sorted(set(unigrams + bigrams))



def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class LightweightRouterModel:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.thresholds = artifact.get("thresholds", {})

    @classmethod
    def load(cls, model_dir: str | Path) -> "LightweightRouterModel":
        model_path = Path(model_dir) / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"Routing model artifact not found at {model_path}. "
                "Run `python scripts/build_router_baseline_artifact.py` first."
            )
        with model_path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def _score_labels(self, features: list[str], section: dict[str, Any]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for label, payload in section.items():
            score = float(payload.get("bias", 0.0))
            weights = payload.get("weights", {})
            for feature in features:
                score += float(weights.get(feature, 0.0))
            scores[label] = _sigmoid(score)
        return scores

    def predict_one(self, text: str) -> dict[str, Any]:
        features = preprocess_text(text)
        confidence: dict[str, Any] = {}
        prediction: dict[str, Any] = {}

        binary_scores = self._score_labels(features, self.artifact["binary_models"])
        for label in BINARY_LABELS:
            confidence[label] = round(binary_scores[label], 4)
            threshold = float(self.thresholds.get("binary", {}).get(label, 0.5))
            prediction[label] = binary_scores[label] >= threshold

        for field, labels in MULTILABEL_FIELDS.items():
            scores = self._score_labels(features, self.artifact["multilabel_models"][field])
            confidence[field] = {label: round(scores[label], 4) for label in labels}
            threshold = float(self.thresholds.get(field, 0.5))
            selected = [label for label in labels if scores[label] >= threshold]
            if field == "crops":
                selected = [max(labels, key=lambda label: scores[label])]
                if prediction["is_ag_related"] and not selected:
                    selected = ["unknown"]
            elif field == "topics":
                if prediction["is_ag_related"] and not selected:
                    selected = ["general_agronomy"]
            prediction[field] = selected

        for field, labels in MULTICLASS_FIELDS.items():
            scores = self._score_labels(features, self.artifact["multiclass_models"][field])
            confidence[field] = {label: round(scores[label], 4) for label in labels}
            prediction[field] = max(labels, key=lambda label: scores[label])

        heuristic = _heuristic_overrides(text)
        if heuristic["is_ag_related"]:
            prediction["is_ag_related"] = True
            confidence["is_ag_related"] = max(confidence["is_ag_related"], 0.85)
            for flag in ROUTING_FLAGS:
                if heuristic[flag]:
                    prediction[flag] = True
                    confidence[flag] = max(confidence[flag], 0.8)
            if heuristic["crops"]:
                prediction["crops"] = heuristic["crops"]
                for crop in heuristic["crops"]:
                    confidence["crops"][crop] = max(confidence["crops"].get(crop, 0.0), 0.82)
            if heuristic["topics"]:
                prediction["topics"] = sorted(set(prediction["topics"]) | set(heuristic["topics"]))
                for topic in heuristic["topics"]:
                    confidence["topics"][topic] = max(confidence["topics"].get(topic, 0.0), 0.78)
            if heuristic["intent"] != "other":
                prediction["intent"] = heuristic["intent"]
                confidence["intent"][heuristic["intent"]] = max(confidence["intent"].get(heuristic["intent"], 0.0), 0.72)
            prediction["urgency"] = heuristic["urgency"]
            confidence["urgency"][heuristic["urgency"]] = max(confidence["urgency"].get(heuristic["urgency"], 0.0), 0.72)
        else:
            prediction["is_ag_related"] = False
            confidence["is_ag_related"] = min(confidence["is_ag_related"], 0.2)
            prediction["crops"] = []
            prediction["topics"] = []
            for flag in ROUTING_FLAGS:
                prediction[flag] = False
                confidence[flag] = min(confidence[flag], 0.2)
            prediction["urgency"] = "low"

        return {"prediction": prediction, "confidence": confidence, "features": features}


@lru_cache(maxsize=1)
def load_router_model(model_dir: str | Path = DEFAULT_MODEL_DIR) -> LightweightRouterModel:
    return LightweightRouterModel.load(model_dir)



def predict_query(text: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Query text must not be empty.")
    model = load_router_model(model_dir)
    result = model.predict_one(cleaned_text)
    return {
        "input": cleaned_text,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "model": {
            "model_dir": str(Path(model_dir)),
            "model_type": "lightweight_token_log_odds",
            "preprocessing": "lowercase + unigram/bigram set",
        },
    }



def route_query(text: str) -> dict[str, Any]:
    """
    Returns routing decision for a given query.
    """
    result = predict_query(text, model_dir=DEFAULT_MODEL_DIR)
    return {"input": result["input"], "prediction": result["prediction"]}



def append_log_record(record: dict[str, Any], log_path: str | Path = DEFAULT_LOG_PATH) -> dict[str, Any]:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record



def log_prediction(result: dict[str, Any], log_path: str | Path = DEFAULT_LOG_PATH) -> dict[str, Any]:
    interaction_id = uuid.uuid4().hex
    record = {
        "id": interaction_id,
        "interaction_id": interaction_id,
        "entry_type": "inference",
        "timestamp": _utc_timestamp(),
        "input": result["input"],
        "prediction": deepcopy(result["prediction"]),
        "confidence": deepcopy(result["confidence"]),
        "user_feedback": None,
        "corrected_labels": None,
    }
    return append_log_record(record, log_path=log_path)



def append_review(
    interaction_id: str,
    user_feedback: str,
    corrected_labels: dict[str, Any] | None = None,
    review_notes: str | None = None,
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    record = {
        "id": uuid.uuid4().hex,
        "interaction_id": interaction_id,
        "entry_type": "review",
        "timestamp": _utc_timestamp(),
        "input": None,
        "prediction": None,
        "confidence": None,
        "user_feedback": user_feedback,
        "corrected_labels": corrected_labels,
        "review_notes": review_notes,
    }
    return append_log_record(record, log_path=log_path)



def read_log_events(log_path: str | Path = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events



def _review_sort_key(event: dict[str, Any]) -> tuple[str, str]:
    return (str(event.get("timestamp", "")), str(event.get("id", "")))



def load_logged_interactions(log_path: str | Path = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    events = read_log_events(log_path)
    interactions: dict[str, dict[str, Any]] = {}
    reviews_by_interaction: dict[str, list[dict[str, Any]]] = {}

    for event in events:
        interaction_id = event.get("interaction_id") or event.get("id")
        if event.get("entry_type") == "inference":
            interactions[interaction_id] = {
                "id": event["id"],
                "interaction_id": interaction_id,
                "timestamp": event["timestamp"],
                "input": event["input"],
                "prediction": deepcopy(event["prediction"]),
                "confidence": deepcopy(event["confidence"]),
                "user_feedback": event.get("user_feedback"),
                "corrected_labels": deepcopy(event.get("corrected_labels")),
                "reviews": [],
            }
        elif event.get("entry_type") == "review":
            reviews_by_interaction.setdefault(interaction_id, []).append(event)

    for interaction_id, interaction in interactions.items():
        for review in sorted(reviews_by_interaction.get(interaction_id, []), key=_review_sort_key):
            interaction["user_feedback"] = review.get("user_feedback")
            if review.get("corrected_labels") is not None:
                interaction["corrected_labels"] = deepcopy(review["corrected_labels"])
            interaction["reviews"].append(review)

    return sorted(interactions.values(), key=lambda row: (row["timestamp"], row["interaction_id"]))



def resolved_labels(interaction: dict[str, Any]) -> dict[str, Any] | None:
    if interaction.get("corrected_labels"):
        return deepcopy(interaction["corrected_labels"])
    if interaction.get("user_feedback") == "correct":
        return deepcopy(interaction["prediction"])
    return None
