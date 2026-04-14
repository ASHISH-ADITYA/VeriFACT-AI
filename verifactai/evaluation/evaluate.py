#!/usr/bin/env python3
"""
VeriFactAI Evaluation Suite.

Two evaluation modes (per Report-Claim Policy P3):

  CORE (fixed-response):
    Verifies pre-existing known-true and known-false text.
    No live LLM generation → deterministic, reproducible.
    These metrics go in the abstract / results tables.

  STRESS TEST (live-generation):
    Uses verify_query() with live LLM generation.
    Results include generation variability.
    Labeled as "Live Generation Stress Test" in appendix.

Usage:
    python evaluation/evaluate.py                              # full suite
    python evaluation/evaluate.py --benchmark truthfulqa-fixed # core only
    python evaluation/evaluate.py --benchmark sanity           # quick check
    python evaluation/evaluate.py --max-samples 100            # limit samples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtime_safety import apply_runtime_safety_defaults

apply_runtime_safety_defaults()

from config import Config, Profile  # noqa: E402
from core.pipeline import VeriFactPipeline  # noqa: E402
from evaluation.domain_benchmark import evaluate_domain_dataset  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    binary_hallucination_metrics,
    latency_percentiles,
    retrieval_recall_at_k,
)
from evaluation.plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_factuality_distribution,
    plot_roc_curve,
    plot_verdict_breakdown,
)

RESULTS_DIR = Path("assets/evaluation")


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# CORE BENCHMARK: TruthfulQA Fixed-Response (no generation variability)
# ═══════════════════════════════════════════════════════════════════════


def evaluate_truthfulqa_fixed(
    pipeline: VeriFactPipeline,
    max_samples: int | None = None,
    threshold: float | None = None,
) -> dict:
    """
    Core benchmark: verify pre-existing known-false answers from TruthfulQA.

    No LLM generation involved — verifies the dataset's incorrect_answers
    against the pipeline's retrieval + NLI components.
    Deterministic and reproducible.
    """
    from datasets import load_dataset

    print("\n=== [CORE] TruthfulQA Fixed-Response Evaluation ===")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    hal_threshold = threshold or pipeline.config.confidence.hallucination_threshold
    print(f"Samples: {len(ds)}, Hallucination threshold: {hal_threshold}")

    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []
    latencies: list[float] = []
    retrieval_hits: list[bool] = []

    for item in tqdm(ds, desc="TruthfulQA-fixed"):
        best_answer = item.get("best_answer", "")
        incorrect_answers = item.get("incorrect_answers", [])
        if not best_answer or not incorrect_answers:
            continue

        # Verify correct answer
        try:
            t0 = time.perf_counter()
            correct_result = pipeline.verify_text(best_answer)
            latencies.append(time.perf_counter() - t0)

            y_true.append(0)  # correct
            flagged = correct_result.factuality_score < (hal_threshold * 100)
            y_pred.append(1 if flagged else 0)
            y_scores.append(1 - correct_result.factuality_score / 100)
            retrieval_hits.extend(bool(c.evidence) for c in correct_result.claims)
        except Exception as exc:
            print(f"  Skip correct ({exc})")
            continue

        # Verify first incorrect answer
        wrong_answer = (
            incorrect_answers[0] if isinstance(incorrect_answers, list) else str(incorrect_answers)
        )
        try:
            t0 = time.perf_counter()
            wrong_result = pipeline.verify_text(wrong_answer)
            latencies.append(time.perf_counter() - t0)

            y_true.append(1)  # hallucinated
            flagged = wrong_result.factuality_score < (hal_threshold * 100)
            y_pred.append(1 if flagged else 0)
            y_scores.append(1 - wrong_result.factuality_score / 100)
            retrieval_hits.extend(bool(c.evidence) for c in wrong_result.claims)
        except Exception as exc:
            print(f"  Skip incorrect ({exc})")
            continue

    if not y_true:
        return {"benchmark": "truthfulqa_fixed", "error": "no_valid_samples"}

    metrics = binary_hallucination_metrics(y_true, y_pred, y_scores)
    metrics["benchmark"] = "truthfulqa_fixed"
    metrics["benchmark_type"] = "CORE"
    metrics["samples"] = len(y_true)
    metrics["latency"] = latency_percentiles(latencies)
    metrics["retrieval_recall_at_k"] = retrieval_recall_at_k(retrieval_hits)

    # Plots
    plot_confusion_matrix(y_true, y_pred, str(RESULTS_DIR / "truthfulqa_fixed_cm.png"))
    if "roc_curve" in metrics:
        plot_roc_curve(
            metrics["roc_curve"]["fpr"],
            metrics["roc_curve"]["tpr"],
            metrics["auroc"],
            str(RESULTS_DIR / "truthfulqa_fixed_roc.png"),
        )

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# STRESS TEST: TruthfulQA Live Generation
# ═══════════════════════════════════════════════════════════════════════


def evaluate_truthfulqa_live(pipeline: VeriFactPipeline, max_samples: int | None = None) -> dict:
    """
    Stress test: ask LLM questions from TruthfulQA, then verify answers.

    INCLUDES generation variability. Results labeled as stress test.
    Uses strict=True (eval mode: primary provider only).
    """
    from datasets import load_dataset

    print("\n=== [STRESS TEST] TruthfulQA Live Generation ===")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"Samples: {len(ds)}")

    factuality_scores: list[float] = []
    latencies: list[float] = []
    total_s = total_c = total_u = total_ne = 0

    for item in tqdm(ds, desc="TruthfulQA-live"):
        try:
            t0 = time.perf_counter()
            result = pipeline.verify_query(item["question"], strict=True)
            latencies.append(time.perf_counter() - t0)

            factuality_scores.append(result.factuality_score)
            total_s += result.supported
            total_c += result.contradicted
            total_u += result.unverifiable
            total_ne += result.no_evidence
        except Exception as exc:
            print(f"  Skip ({exc})")
            continue

    import numpy as np

    metrics = {
        "benchmark": "truthfulqa_live",
        "benchmark_type": "STRESS_TEST",
        "samples": len(factuality_scores),
        "mean_factuality": float(np.mean(factuality_scores)) if factuality_scores else 0,
        "median_factuality": float(np.median(factuality_scores)) if factuality_scores else 0,
        "std_factuality": float(np.std(factuality_scores)) if factuality_scores else 0,
        "total_claims": total_s + total_c + total_u + total_ne,
        "supported": total_s,
        "contradicted": total_c,
        "unverifiable": total_u,
        "no_evidence": total_ne,
        "latency": latency_percentiles(latencies),
    }

    if factuality_scores:
        plot_factuality_distribution(
            factuality_scores, str(RESULTS_DIR / "truthfulqa_live_distribution.png")
        )
        plot_verdict_breakdown(
            total_s,
            total_c,
            total_u,
            total_ne,
            str(RESULTS_DIR / "truthfulqa_live_verdicts.png"),
        )

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# CORE BENCHMARK: HaluEval Fixed-Response
# ═══════════════════════════════════════════════════════════════════════


def evaluate_halueval(
    pipeline: VeriFactPipeline,
    max_samples: int | None = 500,
    threshold: float | None = None,
) -> dict:
    """Core benchmark: distinguish hallucinated vs correct answers in HaluEval."""
    from datasets import load_dataset

    print("\n=== [CORE] HaluEval Fixed-Response Evaluation ===")
    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception:
        print("  HaluEval dataset unavailable — skipping.")
        return {"benchmark": "halueval", "error": "dataset_unavailable"}

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    hal_threshold = threshold or pipeline.config.confidence.hallucination_threshold
    print(f"Samples: {len(ds)}, Threshold: {hal_threshold}")

    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []
    latencies: list[float] = []

    for item in tqdm(ds, desc="HaluEval"):
        # HaluEval schema: answer + hallucination ("yes"/"no") + knowledge
        answer_text = item.get("answer", "")
        is_hallucinated = item.get("hallucination", "").lower().strip() == "yes"
        if not answer_text or len(answer_text) < 20:
            continue

        try:
            t0 = time.perf_counter()
            result = pipeline.verify_text(answer_text)
            latencies.append(time.perf_counter() - t0)

            y_true.append(1 if is_hallucinated else 0)
            flagged = result.factuality_score < (hal_threshold * 100)
            y_pred.append(1 if flagged else 0)
            y_scores.append(1 - result.factuality_score / 100)
        except Exception as exc:
            print(f"  Skip ({exc})")
            continue

    if not y_true:
        return {"benchmark": "halueval", "error": "no_valid_samples"}

    metrics = binary_hallucination_metrics(y_true, y_pred, y_scores)
    metrics["benchmark"] = "halueval"
    metrics["benchmark_type"] = "CORE"
    metrics["samples"] = len(y_true)
    metrics["latency"] = latency_percentiles(latencies)

    plot_confusion_matrix(y_true, y_pred, str(RESULTS_DIR / "halueval_cm.png"))
    if "roc_curve" in metrics:
        plot_roc_curve(
            metrics["roc_curve"]["fpr"],
            metrics["roc_curve"]["tpr"],
            metrics["auroc"],
            str(RESULTS_DIR / "halueval_roc.png"),
        )

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════════


def quick_sanity_check(pipeline: VeriFactPipeline) -> dict:
    """Smoke test with hand-crafted inputs. No LLM generation needed."""
    print("\n=== Quick Sanity Check ===")
    test_cases = [
        {
            "text": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
            "expected": "SUPPORTED",
        },
        {"text": "Albert Einstein invented the telephone in 1920.", "expected": "CONTRADICTED"},
        {"text": "Water boils at 100 degrees Celsius at sea level.", "expected": "SUPPORTED"},
    ]

    results = []
    for tc in test_cases:
        try:
            result = pipeline.verify_text(tc["text"])
            verdicts = [c.verdict for c in result.claims]
            results.append(
                {
                    "text": tc["text"],
                    "expected": tc["expected"],
                    "factuality_score": result.factuality_score,
                    "verdicts": verdicts,
                    "time": result.processing_time,
                }
            )
            print(f"  Score={result.factuality_score:.0f}  {verdicts}  [{tc['text'][:50]}…]")
        except Exception as exc:
            results.append({"text": tc["text"], "error": str(exc)})
            print(f"  ERROR: {exc}")

    return {"benchmark": "sanity_check", "results": results}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriFactAI Evaluation Suite")
    parser.add_argument(
        "--benchmark",
        choices=[
            "truthfulqa-fixed",
            "truthfulqa-live",
            "halueval",
            "domain",
            "sanity",
            "core",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--domain-dataset",
        default=str(PROJECT_ROOT / "evaluation" / "data" / "legal_domain.jsonl"),
        help="JSONL dataset for domain benchmark",
    )
    parser.add_argument(
        "--profile",
        choices=["interactive", "eval"],
        default="eval",
        help="Performance profile (eval recommended for benchmarks)",
    )
    args = parser.parse_args()

    _ensure_dirs()

    profile = Profile.EVAL if args.profile == "eval" else Profile.INTERACTIVE
    pipeline = VeriFactPipeline(Config(), profile=profile)

    all_metrics: dict = {}

    if args.benchmark in ("sanity", "all"):
        all_metrics["sanity"] = quick_sanity_check(pipeline)

    if args.benchmark in ("truthfulqa-fixed", "core", "all"):
        all_metrics["truthfulqa_fixed"] = evaluate_truthfulqa_fixed(pipeline, args.max_samples)

    if args.benchmark in ("halueval", "core", "all"):
        all_metrics["halueval"] = evaluate_halueval(pipeline, args.max_samples)

    if args.benchmark in ("domain", "core", "all"):
        all_metrics["domain"] = evaluate_domain_dataset(
            pipeline,
            dataset_path=args.domain_dataset,
        )

    if args.benchmark in ("truthfulqa-live", "all"):
        all_metrics["truthfulqa_live"] = evaluate_truthfulqa_live(pipeline, args.max_samples)

    # Save
    out_path = RESULTS_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nAll metrics saved to {out_path}")


if __name__ == "__main__":
    main()
