"""
Shared test fixtures for VeriFactAI test suite.

Provides mock objects so tests run without Ollama, FAISS index, or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.claim_decomposer import Claim
from core.evidence_retriever import Evidence


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_claims() -> List[Claim]:
    """Three claims with diverse types for testing."""
    return [
        Claim(
            id="c-0",
            text="The Eiffel Tower is located in Paris, France.",
            source_sentence="The Eiffel Tower is located in Paris, France.",
            claim_type="entity_fact",
            char_start=0,
            char_end=46,
        ),
        Claim(
            id="c-1",
            text="It was completed in 1889.",
            source_sentence="It was completed in 1889.",
            claim_type="temporal",
            char_start=47,
            char_end=71,
        ),
        Claim(
            id="c-2",
            text="Albert Einstein invented the telephone.",
            source_sentence="Albert Einstein invented the telephone.",
            claim_type="entity_fact",
            char_start=72,
            char_end=111,
        ),
    ]


@pytest.fixture
def sample_evidence() -> List[Evidence]:
    """Evidence passages for testing verdict engine."""
    return [
        Evidence(
            text="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            source="wikipedia",
            title="Eiffel Tower",
            url="https://simple.wikipedia.org/wiki/Eiffel_Tower",
            similarity=0.82,
            chunk_id=100,
        ),
        Evidence(
            text="The tower was built from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",
            source="wikipedia",
            title="Eiffel Tower",
            url="https://simple.wikipedia.org/wiki/Eiffel_Tower",
            similarity=0.75,
            chunk_id=101,
        ),
        Evidence(
            text="Alexander Graham Bell is credited with inventing the first practical telephone in 1876.",
            source="wikipedia",
            title="Telephone",
            url="https://simple.wikipedia.org/wiki/Telephone",
            similarity=0.68,
            chunk_id=200,
        ),
    ]


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns deterministic JSON claims."""
    client = MagicMock()
    client.generate.return_value = (
        '[{"claim": "Paris is the capital of France.", '
        '"source_sentence": "Paris is the capital of France.", '
        '"claim_type": "entity_fact"}]'
    )
    return client


@pytest.fixture
def mock_llm_client_failing():
    """Mock LLM client that always raises (to test fallback)."""
    client = MagicMock()
    client.generate.side_effect = RuntimeError("LLM unavailable")
    return client
