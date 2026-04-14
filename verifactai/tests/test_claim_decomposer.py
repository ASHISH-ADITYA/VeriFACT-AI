"""Tests for ClaimDecomposer — parsing, filtering, fallback, span location."""

from __future__ import annotations

import pytest
from core.claim_decomposer import ClaimDecomposer


class TestJsonParsing:
    """Robust JSON extraction from various LLM output formats."""

    def test_direct_json(self):
        raw = '[{"claim": "X", "source_sentence": "X.", "claim_type": "entity_fact"}]'
        assert len(ClaimDecomposer._parse_json(raw)) == 1

    def test_markdown_code_block(self):
        raw = '```json\n[{"claim": "A", "source_sentence": "A.", "claim_type": "temporal"}]\n```'
        assert ClaimDecomposer._parse_json(raw)[0]["claim"] == "A"

    def test_embedded_in_text(self):
        raw = 'Here:\n[{"claim": "B", "source_sentence": "B.", "claim_type": "causal"}]\nDone.'
        assert ClaimDecomposer._parse_json(raw)[0]["claim"] == "B"

    def test_multiple_claims(self):
        raw = '[{"claim": "A", "source_sentence": "A.", "claim_type": "entity_fact"}, {"claim": "B", "source_sentence": "B.", "claim_type": "numerical"}]'
        result = ClaimDecomposer._parse_json(raw)
        assert len(result) == 2
        assert result[1]["claim"] == "B"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            ClaimDecomposer._parse_json("This is not JSON at all")

    def test_empty_array(self):
        assert ClaimDecomposer._parse_json("[]") == []


class TestSentenceFiltering:
    """Tests for _should_skip heuristic."""

    def test_skip_question(self):
        assert ClaimDecomposer._should_skip("Is this a question?") is True

    def test_skip_short(self):
        assert ClaimDecomposer._should_skip("Hi.") is True

    def test_skip_hedge_might(self):
        assert ClaimDecomposer._should_skip("This might possibly be true in some cases.") is True

    def test_skip_hedge_perhaps(self):
        assert ClaimDecomposer._should_skip("Perhaps this could be the answer to the question.") is True

    def test_keep_factual(self):
        assert ClaimDecomposer._should_skip("Albert Einstein was born in Ulm in 1879.") is False

    def test_keep_long_factual(self):
        assert ClaimDecomposer._should_skip("The speed of light in a vacuum is approximately 299,792 kilometers per second.") is False


class TestSpanLocation:
    """Tests for mapping claims back to original text positions."""

    def test_exact_match(self):
        text = "The cat sat on the mat. The dog ran fast."
        start, end = ClaimDecomposer._locate_span(text, "The dog ran fast.")
        assert start == 24
        assert end == 41

    def test_not_found(self):
        start, end = ClaimDecomposer._locate_span("Hello world.", "Not here.")
        assert start == -1

    def test_empty_sentence(self):
        start, end = ClaimDecomposer._locate_span("Some text.", "")
        assert start == -1

    def test_first_sentence(self):
        text = "First sentence. Second sentence."
        start, end = ClaimDecomposer._locate_span(text, "First sentence.")
        assert start == 0
        assert end == 15


class TestDecomposeWithMockLLM:
    """Integration tests using mock LLM client."""

    def test_llm_decompose_success(self, mock_llm_client):
        decomposer = ClaimDecomposer(mock_llm_client)
        claims = decomposer.decompose("Paris is the capital of France.")
        assert len(claims) == 1
        assert claims[0].text == "Paris is the capital of France."
        assert claims[0].id == "c-0"

    def test_fallback_on_llm_failure(self, mock_llm_client_failing):
        decomposer = ClaimDecomposer(mock_llm_client_failing)
        text = "Albert Einstein was born in Ulm in 1879. He won the Nobel Prize in 1921."
        claims = decomposer.decompose(text)
        assert len(claims) >= 1  # spaCy fallback produces at least 1 sentence
        assert all(c.id.startswith("c-") for c in claims)

    def test_empty_input(self, mock_llm_client):
        mock_llm_client.generate.return_value = "[]"
        decomposer = ClaimDecomposer(mock_llm_client)
        claims = decomposer.decompose("")
        # Either LLM returns empty or fallback produces nothing
        assert isinstance(claims, list)
