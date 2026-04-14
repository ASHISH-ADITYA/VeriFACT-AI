"""
Evaluation metrics for VeriFactAI.

Provides:
  - Precision / Recall / F1 for hallucination detection
  - Expected Calibration Error (ECE)
  - Confusion matrix generation
  - ROC / AUC computation
  - Latency percentiles (p50, p95)
  - Retrieval recall@k
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_hallucination_metrics(
    y_true: list[int], y_pred: list[int], y_scores: list[float] | None = None
) -> dict:
    """
    Binary hallucination detection metrics.

    Args:
        y_true: ground truth (1 = hallucinated, 0 = correct)
        y_pred: predictions (1 = flagged, 0 = passed)
        y_scores: continuous scores (higher = more likely hallucinated)
    """
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Correct", "Hallucinated"], output_dict=True
        ),
    }

    if y_scores is not None and len(set(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        result["auroc"] = float(roc_auc_score(y_true, y_scores))
        result["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    return result


def expected_calibration_error(
    confidences: list[float],
    accuracies: list[int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Ref: Guo et al. (2017), "On Calibration of Modern Neural Networks"
    """
    conf = np.array(confidences)
    acc = np.array(accuracies)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    total = len(conf)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        ece += (count / total) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))

    return float(ece)


def latency_percentiles(latencies: list[float]) -> dict:
    """Compute p50 and p95 latency from a list of processing times (seconds)."""
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
    }


def retrieval_recall_at_k(
    relevant_found: list[bool],
) -> float:
    """
    Fraction of claims where at least one relevant evidence passage was retrieved.

    Args:
        relevant_found: per-claim boolean — True if any retrieved evidence
                        had similarity above the relevance threshold.
    """
    if not relevant_found:
        return 0.0
    return float(sum(relevant_found) / len(relevant_found))


def verdict_accuracy(predicted_verdicts: list[str], ground_truth_verdicts: list[str]) -> dict:
    """Multi-class verdict accuracy."""
    label_map = {"SUPPORTED": 0, "CONTRADICTED": 1, "UNVERIFIABLE": 2, "NO_EVIDENCE": 2}
    y_true = [label_map.get(v, 2) for v in ground_truth_verdicts]
    y_pred = [label_map.get(v, 2) for v in predicted_verdicts]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(
            y_true,
            y_pred,
            target_names=["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"],
            output_dict=True,
            zero_division=0,
        ),
    }
