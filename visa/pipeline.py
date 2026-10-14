"""Reproducible binary classification with preprocessing fitted on training data only."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "case_status"
LABELS = {"Denied": 0, "Certified": 1}
DROP = ["case_id"]
DEFAULT_DATA = "data/EasyVisa.csv"


def prepare(frame, training=False):
    frame = frame.drop_duplicates().copy() if training else frame.copy()
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed")
    frame = frame.drop(
        columns=[c for c in frame if c in DROP or c.startswith("Unnamed:")],
        errors="ignore",
    )
    if "TotalCharges" in frame:
        values = frame["TotalCharges"].astype(str).str.strip()
        invalid = pd.to_numeric(values.mask(values.eq(""), np.nan), errors="coerce")
        if (values.ne("") & invalid.isna()).any():
            raise ValueError("TotalCharges must be numeric or blank")
        frame["TotalCharges"] = invalid
    if training:
        if TARGET not in frame:
            raise ValueError(f"Missing target column: {TARGET}")
        if frame[TARGET].isna().any() or not frame[TARGET].isin(LABELS).all():
            raise ValueError(f"{TARGET} must contain only {list(LABELS)}")
        y = frame.pop(TARGET).map(LABELS).astype(int)
        if len(y) < 30 or y.value_counts().reindex([0, 1], fill_value=0).min() < 5:
            raise ValueError("Need at least 30 rows and 5 examples of each class")
    else:
        frame = frame.drop(columns=[TARGET], errors="ignore")
        y = None
    if frame.empty:
        raise ValueError("No input rows or features")
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Numeric features cannot contain infinity")
    return frame, y


def preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X if c not in numeric]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def train(data, output):
    X, y = prepare(pd.read_csv(data), training=True)
    dev_X, test_X, dev_y, test_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_X, val_X, train_y, val_y = train_test_split(
        dev_X, dev_y, test_size=0.25, stratify=dev_y, random_state=42
    )
    classifiers = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    models, scores = {}, {}
    for name, classifier in classifiers.items():
        model = Pipeline(
            [("preprocess", preprocessor(train_X)), ("classifier", classifier)]
        )
        model.fit(train_X, train_y)
        models[name] = model
        scores[name] = float(
            average_precision_score(val_y, model.predict_proba(val_X)[:, 1])
        )
    best = max(scores, key=scores.get)
    model = models[best]
    precision, recall, thresholds = precision_recall_curve(
        val_y, model.predict_proba(val_X)[:, 1]
    )
    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    threshold = float(thresholds[np.argmax(f1)])
    probability = model.predict_proba(test_X)[:, 1]
    prediction = (probability >= threshold).astype(int)
    metrics = {
        "model": best,
        "seed": 42,
        "validation_average_precision": scores,
        "threshold": threshold,
        "test_precision": float(precision_score(test_y, prediction, zero_division=0)),
        "test_recall": float(recall_score(test_y, prediction, zero_division=0)),
        "test_f1": float(f1_score(test_y, prediction, zero_division=0)),
        "test_average_precision": float(average_precision_score(test_y, probability)),
        "test_roc_auc": float(roc_auc_score(test_y, probability)),
        "confusion_matrix": confusion_matrix(
            test_y, prediction, labels=[0, 1]
        ).tolist(),
        "split_rows": {
            "train": len(train_y),
            "validation": len(val_y),
            "test": len(test_y),
        },
        "positive_label": next(k for k, v in LABELS.items() if v == 1),
        "scikit_learn_version": sklearn.__version__,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": X.columns.tolist(),
            "threshold": threshold,
            "labels": LABELS,
            "metrics": metrics,
        },
        output,
    )
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def predict(data, model_path, output):
    saved = joblib.load(model_path)  # Only load trusted artifacts.
    X, _ = prepare(pd.read_csv(data))
    missing = set(saved["features"]) - set(X.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    probability = saved["model"].predict_proba(X[saved["features"]])[:, 1]
    inverse = {v: k for k, v in saved["labels"].items()}
    results = pd.read_csv(data)
    results["positive_class_score"] = probability
    results["predicted_label"] = [
        inverse[int(p >= saved["threshold"])] for p in probability
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data", default=DEFAULT_DATA)
    fit.add_argument("--output", default="artifacts/model.joblib")
    score = sub.add_parser("predict")
    score.add_argument("--input", required=True)
    score.add_argument("--model", default="artifacts/model.joblib")
    score.add_argument("--output", default="artifacts/predictions.csv")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.data, args.output), indent=2))
    else:
        print(
            f"Scored {len(predict(args.input, args.model, args.output))} rows; wrote {args.output}"
        )


if __name__ == "__main__":
    main()
