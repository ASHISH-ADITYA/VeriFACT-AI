"""
Evaluation visualisation suite for VeriFactAI.

Generates publication-quality plots for the project report:
  - Factuality score distribution
  - Verdict breakdown pie chart
  - Confusion matrix heatmap
  - ROC curve
  - Ablation study bar chart
  - Raw LLM vs VeriFactAI comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Consistent style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. Factuality score distribution ──────────────────────────────────

def plot_factuality_distribution(
    scores: List[float], save_path: str = "assets/factuality_distribution.png"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=20, color="steelblue", edgecolor="black", alpha=0.75)
    ax.axvline(60, color="red", linestyle="--", linewidth=1.2, label="Unreliable threshold")
    ax.set_xlabel("Factuality Score (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Factuality Scores — TruthfulQA")
    ax.legend()
    _save(fig, save_path)


# ── 2. Verdict breakdown ─────────────────────────────────────────────

def plot_verdict_breakdown(
    supported: int,
    contradicted: int,
    unverifiable: int,
    no_evidence: int,
    save_path: str = "assets/verdict_breakdown.png",
) -> None:
    labels = ["Supported", "Contradicted", "Unverifiable", "No Evidence"]
    sizes = [supported, contradicted, unverifiable, no_evidence]
    colours = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        sizes, labels=labels, colors=colours, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 11},
    )
    ax.set_title("Claim Verdict Distribution")
    _save(fig, save_path)


# ── 3. Confusion matrix ──────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: str = "assets/confusion_matrix.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Correct", "Hallucinated"],
        cmap="Blues", ax=ax,
    )
    ax.set_title("Hallucination Detection — Confusion Matrix")
    _save(fig, save_path)


# ── 4. ROC curve ─────────────────────────────────────────────────────

def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    auroc: float,
    save_path: str = "assets/roc_curve.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"VeriFactAI (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Hallucination Detection")
    ax.legend(fontsize=10)
    _save(fig, save_path)


# ── 5. Ablation study ────────────────────────────────────────────────

def plot_ablation(
    labels: List[str],
    f1_scores: List[float],
    save_path: str = "assets/ablation_study.png",
) -> None:
    colours = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"][:len(labels)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, f1_scores, color=colours, edgecolor="black", alpha=0.85)
    ax.set_ylabel("F1 Score")
    ax.set_title("Ablation Study — Component Contribution")
    ax.set_ylim(0, 1.0)
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{score:.3f}", ha="center", va="bottom", fontweight="bold",
        )
    _save(fig, save_path)


# ── 6. Raw LLM vs VeriFactAI comparison ──────────────────────────────

def plot_comparison(
    raw_accuracy: float,
    verifact_accuracy: float,
    save_path: str = "assets/comparison.png",
) -> None:
    methods = ["Raw LLM\n(No Verification)", "VeriFactAI\n(With Verification)"]
    scores = [raw_accuracy, verifact_accuracy]
    colours = ["#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(methods, scores, color=colours, edgecolor="black", width=0.5, alpha=0.85)
    ax.set_ylabel("Factual Accuracy (%)")
    ax.set_title("Factual Accuracy: Raw LLM vs VeriFactAI")
    ax.set_ylim(0, 100)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{score:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold",
        )
    _save(fig, save_path)


# ── 7. Confidence calibration plot ───────────────────────────────────

def plot_calibration(
    confidences: List[float],
    accuracies: List[int],
    n_bins: int = 10,
    save_path: str = "assets/calibration.png",
) -> None:
    """Reliability diagram — perfect calibration lies on the diagonal."""
    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    bin_confs, bin_accs = [], []
    for i in range(n_bins):
        mask = (conf_arr > bin_boundaries[i]) & (conf_arr <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_confs.append(float(conf_arr[mask].mean()))
        bin_accs.append(float(acc_arr[mask].mean()))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(
        bin_confs, bin_accs, width=1 / n_bins * 0.8,
        color="steelblue", edgecolor="black", alpha=0.7, label="VeriFactAI",
    )
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Correct Verdicts")
    ax.set_title("Confidence Calibration (Reliability Diagram)")
    ax.legend()
    _save(fig, save_path)
