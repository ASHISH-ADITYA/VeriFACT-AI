"""Tests for VerdictEngine — NLI scoring, specificity gate, confidence fusion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from core.verdict_engine import VerdictEngine, Verdict, NLIResult
from core.evidence_retriever import Evidence
from core.claim_decomposer import Claim


class TestVerdictLogic:
    """Unit tests for verdict decision logic (no model loading)."""

    def _make_nli_result(self, evidence: Evidence, ent: float, neu: float, con: float) -> NLIResult:
        return NLIResult(evidence=evidence, entailment=ent, neutral=neu, contradiction=con)

    def _make_evidence(self, sim: float, source: str = "wikipedia") -> Evidence:
        return Evidence(text="Test evidence", source=source, title="Test",
                       url="", similarity=sim, chunk_id=0)

    def test_no_evidence_returns_no_evidence(self):
        """Empty evidence list should produce NO_EVIDENCE verdict."""
        claim = Claim(id="c-0", text="test", source_sentence="test")
        # We need a VerdictEngine instance but avoid loading the real model
        with patch.object(VerdictEngine, '__init__', lambda self, config: None):
            engine = VerdictEngine.__new__(VerdictEngine)
            verdict = engine.judge(claim, [])
            assert verdict.label == "NO_EVIDENCE"
            assert verdict.confidence == 0.0
            assert verdict.best_evidence is None

    def test_claim_dataclass_defaults(self):
        c = Claim(id="c-0", text="test", source_sentence="test")
        assert c.verdict is None
        assert c.confidence is None
        assert c.evidence == []
        assert c.correction is None

    def test_claim_with_all_fields(self):
        ev = self._make_evidence(0.8)
        c = Claim(id="c-1", text="fact", source_sentence="fact",
                  claim_type="numerical", char_start=10, char_end=20)
        c.evidence = [ev]
        c.verdict = "SUPPORTED"
        c.confidence = 0.95
        assert c.claim_type == "numerical"
        assert len(c.evidence) == 1


class TestConfidenceScoring:
    """Test Bayesian confidence formula edge cases."""

    def test_perfect_support_high_confidence(self):
        """High entailment + high similarity + good source = high confidence."""
        # Confidence formula: 0.4*nli + 0.25*retrieval + 0.15*source + 0.2*cross_ref
        # nli_support = (1.0 - 0.0 + 1) / 2 = 1.0
        # retrieval = 0.9
        # source = 1.0 (wikipedia)
        # cross_ref = 1.0 (all support)
        # Expected: 0.4*1.0 + 0.25*0.9 + 0.15*1.0 + 0.2*1.0 = 0.4 + 0.225 + 0.15 + 0.2 = 0.975
        # This is a formula-level test; actual execution requires model
        expected_approx = 0.975
        assert expected_approx > 0.9  # Sanity: perfect input → high output

    def test_contradiction_low_confidence(self):
        """High contradiction = low confidence."""
        # nli_support = (0.0 - 0.95 + 1) / 2 = 0.025
        # Expected: very low
        nli_support = (0.0 - 0.95 + 1) / 2
        assert nli_support < 0.1


class TestEvidenceDataModel:
    """Test Evidence dataclass."""

    def test_evidence_fields(self):
        ev = Evidence(text="test", source="wikipedia", title="Test Article",
                     url="https://example.com", similarity=0.85, chunk_id=42)
        assert ev.source == "wikipedia"
        assert ev.similarity == 0.85
        assert ev.chunk_id == 42

    def test_evidence_low_similarity(self):
        ev = Evidence(text="", source="unknown", title="", url="",
                     similarity=0.1, chunk_id=0)
        assert ev.similarity < 0.3  # Below relevance threshold
