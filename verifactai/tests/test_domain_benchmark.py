from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evaluation.domain_benchmark import evaluate_domain_dataset


class _FakePipeline:
    class _Cfg:
        class _Conf:
            hallucination_threshold = 0.5

        confidence = _Conf()

    config = _Cfg()

    def verify_text(self, text: str):
        score = 80.0 if "not" in text.lower() else 20.0
        return SimpleNamespace(factuality_score=score)


def test_domain_benchmark_runs_on_sample_dataset():
    pipeline = _FakePipeline()
    dataset = Path(__file__).resolve().parents[1] / "evaluation" / "data" / "legal_domain.jsonl"
    metrics = evaluate_domain_dataset(pipeline, str(dataset))

    assert metrics["benchmark"] == "domain"
    assert metrics["samples"] > 0
    assert "accuracy" in metrics
