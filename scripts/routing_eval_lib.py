#!/usr/bin/env python3
"""Shared utilities for the CornbeltAI routing evaluation harness."""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "routing"
MODELS_DIR = ROOT / "models" / "baseline"
REPORTS_DIR = ROOT / "reports"
CONFUSION_DIR = REPORTS_DIR / "confusion_matrices"
ERROR_DIR = REPORTS_DIR / "error_analysis"
ANALYSIS_DIR = REPORTS_DIR / "analysis"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
ROUTING_FLAGS = ["needs_rag", "needs_web_search", "needs_weather_data", "needs_farm_data"]
BINARY_LABELS = ["is_ag_related", *ROUTING_FLAGS]
MULTILABEL_FIELDS = ["crops", "topics"]
SINGLE_CLASS_FIELDS = ["intent", "urgency"]
ALL_TARGET_FIELDS = [*BINARY_LABELS, *MULTILABEL_FIELDS, *SINGLE_CLASS_FIELDS]
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
HEURISTIC_KEYWORDS = {
    "needs_web_search": [
        "today",
        "right now",
        "current",
        "latest",
        "cash bid",
        "cash bids",
        "basis",
        "futures",
        "price",
        "prices",
        "regulation",
        "cutoff",
        "deadline",
        "dicamba",
        "news",
        "outbreak",
    ],
    "needs_weather_data": [
        "rain",
        "storm",
        "temperature",
        "temperatures",
        "forecast",
        "humidity",
        "wind",
        "frost",
        "heat",
        "drought",
        "spray window",
        "spray today",
        "tomorrow",
        "field to dry",
        "dry spell",
        "weather",
    ],
    "needs_rag": [
        "disease",
        "fungus",
        "fungal",
        "blight",
        "rot",
        "yellow",
        "deficiency",
        "armyworm",
        "aphid",
        "waterhemp",
        "marestail",
        "foxtail",
        "lambsquarters",
        "weed",
        "pest",
        "soil crusting",
        "side-dress",
        "sidedress",
        "what should i do",
        "next move",
    ],
    "needs_farm_data": [
        "field",
        "farm",
        "quarter",
        "north 80",
        "west farm",
        "soil test",
        "planting date",
        "growth stage",
        "hybrid",
        "variety",
        "yield history",
        "home quarter",
        "river bottom",
        "sandy hill",
        "black dirt",
    ],
}
TOPIC_KEYWORDS = {
    "weather": ["rain", "forecast", "wind", "humidity", "temperature", "frost", "heat", "drought", "weather"],
    "disease": ["disease", "blight", "fungus", "rot", "mold", "lesion"],
    "pest": ["pest", "armyworm", "aphid", "beetle", "worm", "insect"],
    "weed": ["weed", "waterhemp", "foxtail", "marestail", "lambsquarters", "palmer"],
    "nutrient": ["nitrogen", "phosphorus", "potassium", "sulfur", "nutrient", "deficiency", "yellow"],
    "soil": ["soil", "infiltration", "compaction", "crusting", "ph", "cec"],
    "management": ["spray", "planting", "harvest", "side-dress", "sidedress", "manage", "rotation"],
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
    "planning": ["should i", "tomorrow", "before next season", "timing", "window"],
    "monitoring": ["watch", "track", "monitor", "scout", "doing"],
    "information_lookup": ["current", "today", "right now", "lookup", "what is the current"],
    "comparison": ["compare", "versus", "vs", "better"],
    "question": ["what", "why", "how"],
}


@dataclass
class DatasetBundle:
    rows: list[dict[str, Any]]
    texts: list[str]
    labels: dict[str, Any]


@dataclass
class BaselineModel:
    embedder_name: str
    binary_models: dict[str, Any]
    multilabel_models: dict[str, Any]
    multiclass_models: dict[str, Any]
    label_spaces: dict[str, list[str]]
    threshold: float = 0.5

    def predict(self, texts: list[str], embeddings: np.ndarray | None = None) -> dict[str, Any]:
        features = embeddings if embeddings is not None else load_embedder(self.embedder_name).encode(texts, show_progress_bar=False)
        predictions: dict[str, Any] = {}
        probabilities: dict[str, Any] = {}

        for label, model in self.binary_models.items():
            preds = model.predict(features)
            predictions[label] = preds.astype(int)
            if hasattr(model, "predict_proba"):
                probabilities[label] = model.predict_proba(features)[:, 1]

        for label, model in self.multilabel_models.items():
            if hasattr(model, "predict_proba"):
                prob_matrix = np.asarray(model.predict_proba(features))
                probabilities[label] = prob_matrix
                if label == "crops":
                    crop_preds = np.zeros_like(prob_matrix, dtype=int)
                    max_indices = prob_matrix.argmax(axis=1)
                    max_scores = prob_matrix.max(axis=1)
                    for row_index, (max_index, max_score) in enumerate(zip(max_indices, max_scores)):
                        if max_score >= self.threshold:
                            crop_preds[row_index, max_index] = 1
                    predictions[label] = crop_preds
                else:
                    predictions[label] = (prob_matrix >= self.threshold).astype(int)
            else:
                predictions[label] = model.predict(features).astype(int)

        for label, model in self.multiclass_models.items():
            predictions[label] = model.predict(features)
            if hasattr(model, "predict_proba"):
                probabilities[label] = model.predict_proba(features)

        predictions["_probabilities"] = probabilities
        return predictions

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "baseline_model.pkl").open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, model_dir: Path) -> "BaselineModel":
        with (model_dir / "baseline_model.pkl").open("rb") as handle:
            return pickle.load(handle)


_EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}


def load_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def ensure_directories() -> None:
    for path in [MODELS_DIR, REPORTS_DIR, CONFUSION_DIR, ERROR_DIR, ANALYSIS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_split(split: str) -> DatasetBundle:
    rows = read_jsonl(DATASET_DIR / f"{split}.jsonl")
    texts = [row["input"] for row in rows]
    labels = build_label_targets(rows)
    return DatasetBundle(rows=rows, texts=texts, labels=labels)


def load_all_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        rows.extend(read_jsonl(DATASET_DIR / f"{split}.jsonl"))
    return rows


def build_label_targets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    labels["is_ag_related"] = np.array([int(row["labels"]["is_ag_related"]) for row in rows], dtype=int)
    for flag in ROUTING_FLAGS:
        labels[flag] = np.array([int(row["labels"][flag]) for row in rows], dtype=int)
    labels["crops"] = encode_multilabel([row["labels"]["crops"] for row in rows], CROP_LABELS)
    labels["topics"] = encode_multilabel([row["labels"]["topics"] for row in rows], TOPIC_LABELS)
    labels["intent"] = np.array([row["labels"]["intent"] for row in rows], dtype=object)
    labels["urgency"] = np.array([row["labels"]["urgency"] for row in rows], dtype=object)
    return labels


def encode_multilabel(value_lists: list[list[str]], classes: list[str]) -> np.ndarray:
    class_to_idx = {label: index for index, label in enumerate(classes)}
    encoded = np.zeros((len(value_lists), len(classes)), dtype=int)
    for row_index, values in enumerate(value_lists):
        for value in values:
            encoded[row_index, class_to_idx[value]] = 1
    return encoded


def decode_multilabel(matrix: np.ndarray, classes: list[str]) -> list[list[str]]:
    decoded: list[list[str]] = []
    for row in matrix:
        decoded.append([label for label, enabled in zip(classes, row) if int(enabled) == 1])
    return decoded


def generate_embeddings(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    embedder = load_embedder(model_name)
    return np.asarray(embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True))


def save_embeddings(output_path: Path, embeddings: np.ndarray) -> None:
    np.save(output_path, embeddings)


def load_embeddings(cache_path: Path) -> np.ndarray | None:
    if cache_path.exists():
        return np.load(cache_path)
    return None


def build_binary_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)


def build_multilabel_classifier() -> OneVsRestClassifier:
    return OneVsRestClassifier(build_binary_classifier())


def build_multiclass_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE)


def train_baseline_model(train_bundle: DatasetBundle, val_bundle: DatasetBundle | None = None) -> BaselineModel:
    ensure_directories()
    train_cache = MODELS_DIR / "train_embeddings.npy"
    val_cache = MODELS_DIR / "val_embeddings.npy"
    train_embeddings = load_embeddings(train_cache)
    if train_embeddings is None:
        train_embeddings = generate_embeddings(train_bundle.texts)
        save_embeddings(train_cache, train_embeddings)
    if val_bundle is not None:
        val_embeddings = load_embeddings(val_cache)
        if val_embeddings is None:
            val_embeddings = generate_embeddings(val_bundle.texts)
            save_embeddings(val_cache, val_embeddings)

    binary_models: dict[str, Any] = {}
    multilabel_models: dict[str, Any] = {}
    multiclass_models: dict[str, Any] = {}

    for label in BINARY_LABELS:
        model = build_binary_classifier()
        model.fit(train_embeddings, train_bundle.labels[label])
        binary_models[label] = model

    for label in MULTILABEL_FIELDS:
        model = build_multilabel_classifier()
        model.fit(train_embeddings, train_bundle.labels[label])
        multilabel_models[label] = model

    for label in SINGLE_CLASS_FIELDS:
        model = build_multiclass_classifier()
        model.fit(train_embeddings, train_bundle.labels[label])
        multiclass_models[label] = model

    baseline_model = BaselineModel(
        embedder_name=EMBEDDING_MODEL_NAME,
        binary_models=binary_models,
        multilabel_models=multilabel_models,
        multiclass_models=multiclass_models,
        label_spaces={
            "crops": CROP_LABELS,
            "topics": TOPIC_LABELS,
            "intent": INTENT_LABELS,
            "urgency": URGENCY_LABELS,
        },
    )
    baseline_model.save(MODELS_DIR)
    metadata = {
        "embedder_name": EMBEDDING_MODEL_NAME,
        "random_state": RANDOM_STATE,
        "train_size": len(train_bundle.rows),
        "val_size": len(val_bundle.rows) if val_bundle else 0,
        "label_spaces": baseline_model.label_spaces,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return baseline_model


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
    }


def evaluate_predictions(bundle: DatasetBundle, predictions: dict[str, Any]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in BINARY_LABELS:
        metrics[label] = binary_metrics(bundle.labels[label], np.asarray(predictions[label]))
    for label in MULTILABEL_FIELDS:
        metrics[label] = multilabel_metrics(bundle.labels[label], np.asarray(predictions[label]))
    for label in SINGLE_CLASS_FIELDS:
        metrics[label] = multiclass_metrics(bundle.labels[label], np.asarray(predictions[label]))
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def metrics_to_markdown(metrics: dict[str, dict[str, float]]) -> str:
    rows = ["| Label | Accuracy | Precision | Recall | F1 | Micro F1 | Macro F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, values in metrics.items():
        rows.append(
            "| {label} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {micro_f1} | {macro_f1} |".format(
                label=label,
                accuracy=values.get("accuracy", 0.0),
                precision=values.get("precision", 0.0),
                recall=values.get("recall", 0.0),
                f1=values.get("f1", 0.0),
                micro_f1=f"{values.get('micro_f1', 0.0):.3f}" if "micro_f1" in values else "-",
                macro_f1=f"{values.get('macro_f1', 0.0):.3f}" if "macro_f1" in values else "-",
            )
        )
    return "\n".join(rows)


def dataframe_from_predictions(bundle: DatasetBundle, predictions: dict[str, Any], source_name: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    pred_crops = decode_multilabel(np.asarray(predictions["crops"]), CROP_LABELS)
    pred_topics = decode_multilabel(np.asarray(predictions["topics"]), TOPIC_LABELS)
    for index, row in enumerate(bundle.rows):
        record: dict[str, Any] = {
            "split": source_name,
            "id": row["id"],
            "input": row["input"],
            "true_is_ag_related": int(bundle.labels["is_ag_related"][index]),
            "pred_is_ag_related": int(predictions["is_ag_related"][index]),
            "true_crops": row["labels"]["crops"],
            "pred_crops": pred_crops[index],
            "true_topics": row["labels"]["topics"],
            "pred_topics": pred_topics[index],
            "true_intent": row["labels"]["intent"],
            "pred_intent": str(predictions["intent"][index]),
            "true_urgency": row["labels"]["urgency"],
            "pred_urgency": str(predictions["urgency"][index]),
            "topic_count": len(row["labels"]["topics"]),
            "crop_count": len(row["labels"]["crops"]),
        }
        for flag in ROUTING_FLAGS:
            record[f"true_{flag}"] = int(bundle.labels[flag][index])
            record[f"pred_{flag}"] = int(predictions[flag][index])
        records.append(record)
    return pd.DataFrame.from_records(records)


def dataset_overview(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counter = Counter()
    ag_counter = Counter()
    crop_counter = Counter()
    topic_counter = Counter()
    intent_counter = Counter()
    flag_counter = Counter()
    for row in rows:
        labels = row["labels"]
        ag_counter[str(labels["is_ag_related"])] += 1
        for crop in labels["crops"]:
            crop_counter[crop] += 1
        for topic in labels["topics"]:
            topic_counter[topic] += 1
        intent_counter[labels["intent"]] += 1
        for flag in ROUTING_FLAGS:
            flag_counter[flag] += int(labels[flag])
    return {
        "total_examples": len(rows),
        "ag_related": ag_counter["True"],
        "non_ag_related": ag_counter["False"],
        "crop_balance": dict(crop_counter),
        "topic_balance": dict(topic_counter),
        "intent_balance": dict(intent_counter),
        "routing_flag_positive_counts": dict(flag_counter),
    }


def routing_decision_summary(frame: pd.DataFrame) -> dict[str, Any]:
    exact_matches = 0
    over_trigger = 0
    under_trigger = 0
    both_error = 0
    breakdowns: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0, "over": 0, "under": 0, "mixed": 0}))
    for _, row in frame.iterrows():
        true_flags = np.array([row[f"true_{flag}"] for flag in ROUTING_FLAGS], dtype=int)
        pred_flags = np.array([row[f"pred_{flag}"] for flag in ROUTING_FLAGS], dtype=int)
        delta = pred_flags - true_flags
        is_over = bool((delta > 0).any())
        is_under = bool((delta < 0).any())
        if np.array_equal(true_flags, pred_flags):
            exact_matches += 1
            result_key = "correct"
        elif is_over and is_under:
            both_error += 1
            result_key = "mixed"
        elif is_over:
            over_trigger += 1
            result_key = "over"
        else:
            under_trigger += 1
            result_key = "under"

        label_groups = {
            "topic": row["true_topics"] or ["none"],
            "crop": row["true_crops"] or ["none"],
            "intent": [row["true_intent"]],
        }
        for group_name, values in label_groups.items():
            for value in values:
                breakdown = breakdowns[group_name][value]
                breakdown["total"] += 1
                breakdown[result_key] += 1

    total = max(len(frame), 1)
    return {
        "exact_match_rate": exact_matches / total,
        "over_trigger_rate": over_trigger / total,
        "under_trigger_rate": under_trigger / total,
        "mixed_error_rate": both_error / total,
        "breakdown": {
            group_name: {
                key: {
                    **counts,
                    "correct_rate": counts["correct"] / max(counts["total"], 1),
                    "over_rate": counts["over"] / max(counts["total"], 1),
                    "under_rate": counts["under"] / max(counts["total"], 1),
                    "mixed_rate": counts["mixed"] / max(counts["total"], 1),
                }
                for key, counts in values.items()
            }
            for group_name, values in breakdowns.items()
        },
    }


def heuristic_predict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions: dict[str, Any] = {label: [] for label in ALL_TARGET_FIELDS}
    for row in rows:
        text = row["input"].lower()
        is_ag = any(keyword in text for keywords in CROP_KEYWORDS.values() for keyword in keywords)
        is_ag = is_ag or any(keyword in text for values in TOPIC_KEYWORDS.values() for keyword in values)
        is_ag = is_ag or any(keyword in text for values in HEURISTIC_KEYWORDS.values() for keyword in values)
        predictions["is_ag_related"].append(int(is_ag))

        crop_values: list[str] = []
        has_corn = any(keyword in text for keyword in CROP_KEYWORDS["corn"])
        has_soy = any(keyword in text for keyword in CROP_KEYWORDS["soybean"])
        if has_corn and has_soy:
            crop_values = ["both"]
        elif has_corn:
            crop_values = ["corn"]
        elif has_soy:
            crop_values = ["soybean"]
        elif is_ag:
            crop_values = ["unknown"]
        predictions["crops"].append(crop_values)

        topic_values = [topic for topic, keywords in TOPIC_KEYWORDS.items() if any(keyword in text for keyword in keywords)]
        if not topic_values and is_ag:
            topic_values = ["general_agronomy"]
        predictions["topics"].append(topic_values)

        for flag, keywords in HEURISTIC_KEYWORDS.items():
            predictions[flag].append(int(any(keyword in text for keyword in keywords) and is_ag))

        intent = "other"
        for candidate, keywords in INTENT_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                intent = candidate
                break
        if not is_ag and intent == "other":
            intent = "question" if "?" in text else "other"
        predictions["intent"].append(intent)

        urgency = "low"
        if any(keyword in text for keyword in ["now", "today", "tomorrow", "urgent", "right now"]):
            urgency = "high"
        elif any(keyword in text for keyword in ["soon", "next", "this week"]):
            urgency = "medium"
        elif is_ag:
            urgency = "medium"
        predictions["urgency"].append(urgency)

    encoded = {
        "is_ag_related": np.asarray(predictions["is_ag_related"], dtype=int),
        "crops": encode_multilabel(predictions["crops"], CROP_LABELS),
        "topics": encode_multilabel(predictions["topics"], TOPIC_LABELS),
        "intent": np.asarray(predictions["intent"], dtype=object),
        "urgency": np.asarray(predictions["urgency"], dtype=object),
    }
    for flag in ROUTING_FLAGS:
        encoded[flag] = np.asarray(predictions[flag], dtype=int)
    return encoded


def compare_metric_tables(model_metrics: dict[str, dict[str, float]], rule_metrics: dict[str, dict[str, float]], primary_metric: str = "f1") -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for label in model_metrics:
        metric_name = "micro_f1" if label in MULTILABEL_FIELDS else primary_metric
        model_value = model_metrics[label].get(metric_name, 0.0)
        rule_value = rule_metrics[label].get(metric_name, 0.0)
        winner = "tie"
        if model_value > rule_value:
            winner = "model"
        elif rule_value > model_value:
            winner = "rules"
        comparisons.append(
            {
                "label": label,
                "metric": metric_name,
                "model": model_value,
                "rules": rule_value,
                "winner": winner,
                "delta": model_value - rule_value,
            }
        )
    return comparisons
