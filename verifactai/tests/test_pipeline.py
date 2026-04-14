"""
Integration tests for VeriFactAI pipeline components.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.claim_decomposer import ClaimDecomposer, Claim


# ── Claim decomposer tests ────────────────────────────────────────────

class TestClaimDecomposer:
    """Test JSON parsing and span location (no LLM needed)."""

    def test_parse_json_direct(self):
        raw = '[{"claim": "X is Y", "source_sentence": "X is Y.", "claim_type": "entity_fact"}]'
        result = ClaimDecomposer._parse_json(raw)
        assert len(result) == 1
        assert result[0]["claim"] == "X is Y"

    def test_parse_json_from_code_block(self):
        raw = '```json\n[{"claim": "A", "source_sentence": "A.", "claim_type": "entity_fact"}]\n```'
        result = ClaimDecomposer._parse_json(raw)
        assert len(result) == 1

    def test_parse_json_embedded(self):
        raw = 'Here are the claims:\n[{"claim": "B", "source_sentence": "B.", "claim_type": "temporal"}]\nDone.'
        result = ClaimDecomposer._parse_json(raw)
        assert result[0]["claim"] == "B"

    def test_should_skip_question(self):
        assert ClaimDecomposer._should_skip("Is this a question?") is True

    def test_should_skip_short(self):
        assert ClaimDecomposer._should_skip("Too short.") is True

    def test_should_skip_hedge(self):
        assert ClaimDecomposer._should_skip("This might possibly be true in some cases.") is True

    def test_should_not_skip_factual(self):
        assert ClaimDecomposer._should_skip("Albert Einstein was born in Ulm in 1879.") is False

    def test_locate_span_exact(self):
        text = "The cat sat on the mat. The dog ran fast."
        start, end = ClaimDecomposer._locate_span(text, "The dog ran fast.")
        assert start == 24
        assert end == 41

    def test_locate_span_missing(self):
        text = "Hello world."
        start, end = ClaimDecomposer._locate_span(text, "Not in the text at all.")
        assert start == -1


# ── Evidence / Verdict data model tests ───────────────────────────────

class TestDataModels:
    def test_claim_defaults(self):
        c = Claim(id="c-0", text="test", source_sentence="test")
        assert c.verdict is None
        assert c.confidence is None
        assert c.evidence == []

    def test_claim_json_serializable(self):
        c = Claim(id="c-0", text="test", source_sentence="test", claim_type="temporal")
        d = {"id": c.id, "text": c.text, "type": c.claim_type}
        assert json.dumps(d)  # should not raise


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
