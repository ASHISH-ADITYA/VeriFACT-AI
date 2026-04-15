"""Smoke tests — fast sanity checks that run after every file edit via hooks."""

import importlib
import sys
from pathlib import Path

# Ensure verifactai is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "verifactai"))


def test_core_modules_import():
    """All core pipeline modules must import without error."""
    for mod in [
        "core.fact_rules",
        "core.annotator",
        "core.verdict_engine",
        "core.evidence_retriever",
        "core.claim_decomposer",
        "core.pipeline",
    ]:
        importlib.import_module(mod)


def test_fact_rules_no_false_positive():
    """True facts must NOT trigger rule violations."""
    from core.fact_rules import check_rules

    safe_claims = [
        "The Eiffel Tower is in Paris, France.",
        "Water boils at 100 degrees Celsius.",
        "World War II ended in 1945.",
    ]
    for claim in safe_claims:
        assert check_rules(claim) is None, f"False positive on: {claim}"


def test_fact_rules_catch_obvious():
    """Obviously false facts must be caught."""
    from core.fact_rules import check_rules

    assert check_rules("The Great Wall of China is in South America") is not None
    assert check_rules("World War II ended in 1920") is not None
