from __future__ import annotations

from config import Config
from core.evidence_retriever import Evidence
from core.selfcheck import SelfCheckScorer


class _DummyLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.idx = 0

    def generate(self, **kwargs):
        if self.idx >= len(self.outputs):
            return self.outputs[-1]
        out = self.outputs[self.idx]
        self.idx += 1
        return out


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            text="The Eiffel Tower is located in Paris and opened in 1889.",
            source="wikipedia",
            title="Eiffel Tower",
            url="",
            similarity=0.88,
            chunk_id=1,
        )
    ]


def test_selfcheck_returns_metrics_with_semantic_entropy():
    outputs = [
        '{"label":"supported","rationale":"Evidence states Paris and opening year."}',
        '{"label":"supported","rationale":"The passage directly confirms the claim."}',
        '{"label":"uncertain","rationale":"Some details are not explicit."}',
        '{"label":"supported","rationale":"The evidence supports this statement."}',
        '{"label":"supported","rationale":"Claim matches evidence facts."}',
    ]
    cfg = Config()
    cfg.selfcheck.enabled = True
    cfg.selfcheck.samples = 5
    cfg.selfcheck.min_valid_samples = 3

    scorer = SelfCheckScorer(_DummyLLM(outputs), cfg)
    result = scorer.score_claim("The Eiffel Tower opened in 1889.", _evidence())

    assert result is not None
    assert result["valid_samples"] >= 3
    assert "semantic_cluster_entropy" in result
    assert 0.0 <= result["semantic_cluster_entropy"] <= 1.0
    assert 0.0 <= result["uncertainty"] <= 1.0
    assert 0.0 <= result["stability"] <= 1.0


def test_selfcheck_none_when_too_few_valid_samples():
    outputs = [
        "invalid",
        "still invalid",
    ]
    cfg = Config()
    cfg.selfcheck.enabled = True
    cfg.selfcheck.samples = 2
    cfg.selfcheck.min_valid_samples = 2

    scorer = SelfCheckScorer(_DummyLLM(outputs), cfg)
    result = scorer.score_claim("Claim text", _evidence())

    assert result is None
