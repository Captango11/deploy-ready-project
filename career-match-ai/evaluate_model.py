"""Evaluate the trained model on the held-out 20% test split.
Usage: python evaluate_model.py
"""
import json
import os
import sys
from datetime import datetime, timezone

import joblib
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from train_model import MODEL_DIR, get_split, load_data


def evaluate():
    clf = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
    vec = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    df = load_data()
    _, X_test, _, y_test = get_split(df)  # same seed + stratify => same held-out rows
    y_pred = clf.predict(vec.transform(X_test))
    labels = list(clf.classes_)

    metrics = {
        "evaluated_on": "held-out test data (20%, stratified, random_state=42)",
        "n_test": int(len(y_test)),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "per_class": {k: v for k, v in classification_report(
            y_test, y_pred, output_dict=True, zero_division=0).items() if k in labels},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Accuracy {metrics['accuracy']:.4f} | Macro F1 {metrics['f1_macro']:.4f}")
    return metrics


if __name__ == "__main__":
    try:
        evaluate()
    except FileNotFoundError:
        print("Model files missing. Run: python train_model.py")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)
