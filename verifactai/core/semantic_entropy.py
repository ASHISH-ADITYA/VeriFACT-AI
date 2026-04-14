"""
Semantic entropy utilities for sampled verification judgements.

Implements normalized entropy and a disagreement proxy used to quantify
uncertainty from multiple stochastic checks.
"""

from __future__ import annotations

from collections import Counter
from math import log
import re


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


def cluster_texts_jaccard(texts: list[str], threshold: float = 0.5) -> list[list[str]]:
    """Cluster texts with a simple token Jaccard threshold."""
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return []

    clusters: list[tuple[set[str], list[str]]] = []
    for text in cleaned:
        tokens = _token_set(text)
        if not tokens:
            continue

        assigned = False
        for idx, (centroid, items) in enumerate(clusters):
            score = _jaccard(tokens, centroid)
            if score >= threshold:
                items.append(text)
                clusters[idx] = (centroid | tokens, items)
                assigned = True
                break

        if not assigned:
            clusters.append((set(tokens), [text]))

    return [items for _, items in clusters]


def cluster_entropy(texts: list[str], threshold: float = 0.5) -> float:
    """Compute normalized entropy over semantic clusters of sampled texts."""
    clusters = cluster_texts_jaccard(texts, threshold=threshold)
    if not clusters:
        return 1.0

    sizes = [len(c) for c in clusters]
    total = float(sum(sizes))
    probs = [s / total for s in sizes if s > 0]
    if not probs:
        return 1.0

    entropy = -sum(p * log(p) for p in probs)
    max_entropy = log(max(len(probs), 2))
    if max_entropy <= 0:
        return 0.0
    return min(max(entropy / max_entropy, 0.0), 1.0)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)
