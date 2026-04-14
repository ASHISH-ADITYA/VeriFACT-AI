"""Tests for AnnotatedOutputGenerator — HTML, JSON, corrections."""

from __future__ import annotations

from core.annotator import AnnotatedOutputGenerator
from core.claim_decomposer import Claim
from core.evidence_retriever import Evidence


class TestJsonReport:
    """Test structured JSON output generation."""

    def test_empty_claims(self):
        gen = AnnotatedOutputGenerator()
        result = gen.generate_json("Some text.", [])
        assert result["total_claims"] == 0
        assert result["factuality_score"] == 0.0
        assert result["original_text"] == "Some text."

    def test_single_supported_claim(self):
        claim = Claim(id="c-0", text="Paris is in France.", source_sentence="Paris is in France.",
                     char_start=0, char_end=19)
        claim.verdict = "SUPPORTED"
        claim.confidence = 0.9
        claim.best_evidence = Evidence(text="ev", source="wikipedia", title="T",
                                       url="", similarity=0.8, chunk_id=0)
        claim.nli_scores = {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}

        gen = AnnotatedOutputGenerator()
        result = gen.generate_json("Paris is in France.", [claim])
        assert result["total_claims"] == 1
        assert result["supported"] == 1
        assert result["factuality_score"] == 100.0
        assert result["claims"][0]["verdict"] == "SUPPORTED"
        assert "evidence" in result["claims"][0]

    def test_mixed_verdicts(self, sample_claims):
        sample_claims[0].verdict = "SUPPORTED"
        sample_claims[0].confidence = 0.9
        sample_claims[1].verdict = "UNVERIFIABLE"
        sample_claims[1].confidence = 0.4
        sample_claims[2].verdict = "CONTRADICTED"
        sample_claims[2].confidence = 0.2

        gen = AnnotatedOutputGenerator()
        original = "The Eiffel Tower is located in Paris, France. It was completed in 1889. Albert Einstein invented the telephone."
        result = gen.generate_json(original, sample_claims)
        assert result["total_claims"] == 3
        assert result["supported"] == 1
        assert result["unverifiable"] == 1
        assert result["contradicted"] == 1
        assert 30 < result["factuality_score"] < 40  # 1/3 supported


class TestHtmlAnnotation:
    """Test HTML output for display."""

    def test_empty_claims_returns_escaped_text(self):
        gen = AnnotatedOutputGenerator()
        html = gen.generate_html("Hello <world>", [])
        assert "&lt;" in html  # HTML escaped
        assert "<script>" not in html  # No injection

    def test_supported_claim_gets_green(self):
        claim = Claim(id="c-0", text="Fact.", source_sentence="Fact.",
                     char_start=0, char_end=5)
        claim.verdict = "SUPPORTED"
        claim.confidence = 0.9
        claim.best_evidence = None

        gen = AnnotatedOutputGenerator()
        html = gen.generate_html("Fact.", [claim])
        assert "verified" in html or "#d4edda" in html

    def test_xss_prevention(self):
        """Ensure user input is HTML-escaped in output."""
        gen = AnnotatedOutputGenerator()
        html = gen.generate_html('<script>alert("xss")</script>', [])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
