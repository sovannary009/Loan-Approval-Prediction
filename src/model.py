"""
model.py
--------
Random Forest model training, cross-validation, and evaluation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
)


def build_model(
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Return a configured RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def cross_validate(model, X, y, n_splits: int = 5) -> dict:
    """Return dict of CV metrics (accuracy, roc_auc, f1)."""
    skf = StratifiedKFold(n_splits=n_splits)
    metrics = {}
    for scoring in ["accuracy", "roc_auc", "f1"]:
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        metrics[scoring] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    return metrics


def evaluate(model, X_val, y_val) -> dict:
    """Compute validation-set metrics and return prediction arrays."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_prob),
        "f1": f1_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred, target_names=["Rejected", "Approved"]),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "roc_curve": roc_curve(y_val, y_prob),
    }


def feature_importance_df(model, feature_names: list) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
