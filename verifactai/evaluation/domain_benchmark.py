"""
Domain benchmark utilities for legal/real-world style hallucination profiling.

Expected JSONL schema per line:
{
  "text": "model answer text",
  "label": 0|1
}
Where label=1 means hallucinated and label=0 means grounded.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.metrics import binary_hallucination_metrics, latency_percentiles


def load_domain_jsonl(path: str) -> tuple[list[str], list[int]]:
    file_path = Path(path)
    if not file_path.exists():
        return ([], [])

    texts: list[str] = []
    labels: list[int] = []
    with file_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            label = int(item.get("label", 0))
            if not text:
                continue
            texts.append(text)
            labels.append(1 if label else 0)

    return (texts, labels)


def evaluate_domain_dataset(
    pipeline,
    dataset_path: str,
    threshold: float | None = None,
) -> dict:
    texts, labels = load_domain_jsonl(dataset_path)
    if not texts:
        return {
            "benchmark": "domain",
            "error": "dataset_unavailable_or_empty",
            "dataset_path": dataset_path,
        }

    hal_threshold = threshold or pipeline.config.confidence.hallucination_threshold

    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []
    latencies: list[float] = []

    import time

    for text, gt in zip(texts, labels, strict=False):
        t0 = time.perf_counter()
        result = pipeline.verify_text(text)
        latencies.append(time.perf_counter() - t0)

        score = 1 - result.factuality_score / 100
        pred = 1 if result.factuality_score < (hal_threshold * 100) else 0

        y_true.append(gt)
        y_pred.append(pred)
        y_scores.append(score)

    metrics = binary_hallucination_metrics(y_true, y_pred, y_scores)
    metrics["benchmark"] = "domain"
    metrics["benchmark_type"] = "CORE"
    metrics["samples"] = len(y_true)
    metrics["dataset_path"] = dataset_path
    metrics["latency"] = latency_percentiles(latencies)
    return metrics
