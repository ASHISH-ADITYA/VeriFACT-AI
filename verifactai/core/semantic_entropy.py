"""
Semantic entropy utilities for sampled verification judgements.

Implements normalized entropy and a disagreement proxy used to quantify
uncertainty from multiple stochastic checks.
"""

from __future__ import annotations

from collections import Counter
from math import log


def normalized_entropy(labels: list[str], classes: list[str]) -> float:
    """Return entropy normalized to [0, 1] across the provided class set."""
    if not labels:
        return 1.0

    counts = Counter(labels)
    total = float(len(labels))

    probs: list[float] = []
    for c in classes:
        p = counts.get(c, 0) / total
        if p > 0:
            probs.append(p)

    if not probs:
        return 1.0

    entropy = -sum(p * log(p) for p in probs)
    max_entropy = log(max(len(classes), 2))
    if max_entropy <= 0:
        return 0.0
    return min(max(entropy / max_entropy, 0.0), 1.0)


def disagreement_ratio(labels: list[str]) -> float:
    """Fraction of samples not in the majority class."""
    if not labels:
        return 1.0

    counts = Counter(labels)
    majority = max(counts.values())
    return min(max(1.0 - (majority / len(labels)), 0.0), 1.0)


def label_distribution(labels: list[str], classes: list[str]) -> dict[str, float]:
    """Class probability distribution for the sampled labels."""
    if not labels:
        return {c: 0.0 for c in classes}

    counts = Counter(labels)
    total = float(len(labels))
    return {c: round(counts.get(c, 0) / total, 4) for c in classes}
