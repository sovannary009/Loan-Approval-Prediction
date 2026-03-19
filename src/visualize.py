"""
visualize.py
------------
Reusable plotting functions for the Loan Approval ML project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

BLUE   = "#2E86AB"
RED    = "#E84855"
DARK   = "#1B4F72"
LIGHT  = "#A8C5DA"
BG     = "#F8F9FA"


def plot_target_distribution(y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    counts = pd.Series(y).map({1: "Approved", 0: "Rejected"}).value_counts()
    bars = ax.bar(counts.index, counts.values, color=[BLUE, RED], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha="center", fontweight="bold")
    ax.set_title("Loan Status Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    return ax


def plot_feature_importance(fi_df, ax=None, top_n=15):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    fi = fi_df.head(top_n)
    colors = [DARK if i < 5 else BLUE if i < 10 else LIGHT for i in range(len(fi))]
    h = ax.barh(fi["Feature"], fi["Importance"], color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_title("Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar, val in zip(h, fi["Importance"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    return ax


def plot_confusion_matrix(cm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Confusion Matrix", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return ax


def plot_roc_curve(fpr, tpr, roc_auc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
    ax.set_title("ROC Curve", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_cv_scores(cv_metrics, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    acc_scores = cv_metrics["accuracy"]["scores"]
    roc_scores = cv_metrics["roc_auc"]["scores"]
    x = np.arange(len(acc_scores))
    w = 0.35
    b1 = ax.bar(x - w / 2, acc_scores, w, color=BLUE,  alpha=0.85, label="Accuracy", edgecolor="white")
    b2 = ax.bar(x + w / 2, roc_scores, w, color=RED,   alpha=0.85, label="ROC-AUC",  edgecolor="white")
    ax.axhline(acc_scores.mean(), color=BLUE, linestyle="--", lw=1.5)
    ax.axhline(roc_scores.mean(), color=RED,  linestyle="--", lw=1.5)
    ax.set_title("5-Fold CV: Accuracy vs ROC-AUC", fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(len(acc_scores))])
    ax.set_ylim(0.5, 1.1)
    ax.legend()
    for bar, val in zip(list(b1) + list(b2), list(acc_scores) + list(roc_scores)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=7.5)
    return ax


def full_report_figure(fi_df, cm, fpr, tpr, roc_auc, cv_metrics, save_path=None):
    """Produce the 4-panel evaluation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Random Forest – Loan Approval Prediction", fontsize=15, fontweight="bold")
    plot_feature_importance(fi_df, ax=axes[0, 0])
    plot_confusion_matrix(cm, ax=axes[0, 1])
    plot_roc_curve(fpr, tpr, roc_auc, ax=axes[1, 0])
    plot_cv_scores(cv_metrics, ax=axes[1, 1])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
