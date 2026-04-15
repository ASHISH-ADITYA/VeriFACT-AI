#!/usr/bin/env python3
"""
Adversarial Benchmark Runner for VeriFACT AI.

Tests the pipeline against obvious false claims, true claims,
and unverifiable claims. Reports per-class precision/recall/F1
with focus on contradiction detection quality.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.runtime_safety import apply_runtime_safety_defaults

apply_runtime_safety_defaults()

from config import Config, Profile
from core.pipeline import VeriFactPipeline


def main() -> None:
    benchmark_path = Path(__file__).parent / "adversarial_benchmark.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    pipeline = VeriFactPipeline(Config(), profile=Profile.INTERACTIVE)

    results = {
        "CONTRADICTED": {"tp": 0, "fp": 0, "fn": 0},
        "SUPPORTED": {"tp": 0, "fp": 0, "fn": 0},
        "UNVERIFIABLE": {"tp": 0, "fp": 0, "fn": 0},
    }
    failures = []
    latencies = []

    print(f"Running {len(benchmark)} claims...")
    print()

    for i, item in enumerate(benchmark):
        claim_text = item["claim"]
        expected = item["label"]

        t0 = time.perf_counter()
        try:
            r = pipeline.verify_text(claim_text)
            if not r.claims:
                actual = "NO_EVIDENCE"
            else:
                # Check ALL sub-claims, not just claims[0].
                # If ANY sub-claim is CONTRADICTED, the input is CONTRADICTED.
                # This handles LLM decomposer splitting "X is in Y" into
                # ["X exists" (SUPPORTED), "X is in Y" (CONTRADICTED)].
                verdicts = [c.verdict for c in r.claims]
                if "CONTRADICTED" in verdicts:
                    actual = "CONTRADICTED"
                elif "SUPPORTED" in verdicts:
                    actual = "SUPPORTED"
                else:
                    actual = verdicts[0]
        except Exception as e:
            actual = "ERROR"
            print(f"  ERROR on claim {i}: {e}")
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # Map NO_EVIDENCE to UNVERIFIABLE for comparison
        if actual == "NO_EVIDENCE":
            actual = "UNVERIFIABLE"

        match = actual == expected
        icon = "PASS" if match else "FAIL"

        if not match:
            failures.append(
                {
                    "claim": claim_text,
                    "expected": expected,
                    "actual": actual,
                    "domain": item.get("domain", ""),
                }
            )

        # Update confusion counts
        for label in results:
            if expected == label and actual == label:
                results[label]["tp"] += 1
            elif actual == label and expected != label:
                results[label]["fp"] += 1
            elif expected == label and actual != label:
                results[label]["fn"] += 1

        status = f"[{icon}]" if not match else "[ OK]"
        print(f"  {status} {expected:14s} → {actual:14s} | {elapsed:.1f}s | {claim_text[:55]}")

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ADVERSARIAL BENCHMARK RESULTS")
    print("=" * 60)
    print()

    total = len(benchmark)
    correct = total - len(failures)
    print(f"  Total claims: {total}")
    print(f"  Correct: {correct}/{total} ({correct / total * 100:.1f}%)")
    print()

    for label, counts in results.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(
            f"  {label:14s}  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})"
        )

    # Contradiction-specific KPIs
    c = results["CONTRADICTED"]
    con_recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0
    con_precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
    false_unverifiable = sum(
        1 for f in failures if f["expected"] == "CONTRADICTED" and f["actual"] == "UNVERIFIABLE"
    )
    total_contradicted = sum(1 for item in benchmark if item["label"] == "CONTRADICTED")

    print()
    print(f"  KEY METRIC: Contradiction Recall = {con_recall:.3f} (target >= 0.90)")
    print(f"  KEY METRIC: Contradiction Precision = {con_precision:.3f} (target >= 0.85)")
    print(
        f"  KEY METRIC: False-UNVERIFIABLE on obvious contradictions = {false_unverifiable}/{total_contradicted} ({false_unverifiable / total_contradicted * 100:.1f}%, target <= 10%)"
    )
    print()

    import numpy as np

    lat = np.array(latencies)
    print(
        f"  Latency: p50={np.percentile(lat, 50):.1f}s  p95={np.percentile(lat, 95):.1f}s  mean={lat.mean():.1f}s"
    )

    if failures:
        print()
        print(f"  FAILURES ({len(failures)}):")
        for f in failures[:15]:
            print(f"    {f['expected']:14s} → {f['actual']:14s} | {f['claim'][:55]}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
