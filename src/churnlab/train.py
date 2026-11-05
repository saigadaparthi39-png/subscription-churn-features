"""Train a logistic baseline. The scaler is fit on the training fold only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churnlab.features import (
    FEATURE_CATALOG,
    MODEL_FEATURES,
    compute_features,
    design_matrix,
    scoring_population,
)


def train(raw_dir: Path, model_path: Path, metrics_path: Path, seed: int = 7) -> dict:
    rows = scoring_population(compute_features(raw_dir))
    features, labels = design_matrix(rows)
    if len(set(labels.tolist())) < 2:
        raise RuntimeError("both classes are required to train")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = LogisticRegression(max_iter=500, random_state=seed)
    model.fit(x_train_scaled, y_train)
    probabilities = model.predict_proba(x_test_scaled)[:, 1]
    metrics = {
        "auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "positive_rate": round(float(labels.mean()), 4),
        "features": FEATURE_CATALOG,
        "model_features": MODEL_FEATURES,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_names": MODEL_FEATURES}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def score(raw_dir: Path, model_path: Path, scores_path: Path) -> int:
    bundle = joblib.load(model_path)
    rows = scoring_population(compute_features(raw_dir))
    features, _ = design_matrix(rows)
    probabilities = bundle["model"].predict_proba(bundle["scaler"].transform(features))[:, 1]
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["account_id", "churn_probability"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            writer.writerow(
                {
                    "account_id": row["account_id"],
                    "churn_probability": f"{float(probability):.6f}",
                }
            )
    return len(rows)
