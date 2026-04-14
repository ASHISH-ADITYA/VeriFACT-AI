#!/usr/bin/env python3
"""
VeriFactAI Smoke Test — one-command local validation.

Checks all prerequisites and runs a minimal end-to-end test.
No paid API keys required. Safe to run on MacBook Air M4.

Usage:
    cd verifactai && python smoke_test.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from utils.runtime_safety import apply_runtime_safety_defaults

apply_runtime_safety_defaults()

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _status(ok: bool) -> str:
    return PASS if ok else FAIL


def check_python_version() -> str:
    v = sys.version_info
    ok = v >= (3, 10)
    print(f"  Python version: {v.major}.{v.minor}.{v.micro} — {_status(ok)}")
    return _status(ok)


def check_spacy_model() -> str:
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print(f"  spaCy en_core_web_sm: {PASS}")
        return PASS
    except Exception:
        print(f"  spaCy en_core_web_sm: {FAIL} — run: python -m spacy download en_core_web_sm")
        return FAIL


def check_ollama() -> str:
    try:
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"  Ollama running: {PASS} — models: {models[:5]}")
        return PASS
    except Exception:
        print(f"  Ollama running: {WARN} — not reachable (spaCy fallback will be used)")
        return WARN


def check_faiss_index() -> str:
    from config import cfg
    p = Path(cfg.retrieval.index_path)
    if p.exists():
        mb = p.stat().st_size / (1024 * 1024)
        print(f"  FAISS index: {PASS} ({mb:.0f} MB at {p})")
        return PASS
    else:
        print(f"  FAISS index: {FAIL} — run: python data/build_index.py --wiki-only --max-articles 5000")
        return FAIL


def check_nli_model() -> str:
    try:
        from transformers import AutoTokenizer
        from config import cfg
        AutoTokenizer.from_pretrained(cfg.nli.model_name)
        print(f"  NLI model ({cfg.nli.model_name}): {PASS}")
        return PASS
    except Exception as exc:
        print(f"  NLI model: {FAIL} — {exc}")
        return FAIL


def check_embedding_model() -> str:
    try:
        from sentence_transformers import SentenceTransformer
        from config import cfg
        SentenceTransformer(cfg.embedding.model_name)
        print(f"  Embedding model ({cfg.embedding.model_name}): {PASS}")
        return PASS
    except Exception as exc:
        print(f"  Embedding model: {FAIL} — {exc}")
        return FAIL


def run_decomposition_test() -> str:
    """Test spaCy fallback decomposition (no LLM needed)."""
    from core.claim_decomposer import ClaimDecomposer

    class _DummyLLM:
        def generate(self, **kwargs):
            raise RuntimeError("Intentional: testing spaCy fallback")

    decomposer = ClaimDecomposer(_DummyLLM())
    text = "Albert Einstein was born in Ulm in 1879. He won the Nobel Prize in Physics in 1921."
    claims = decomposer.decompose(text)
    ok = len(claims) >= 1
    print(f"  Claim decomposition (spaCy fallback): {_status(ok)} — {len(claims)} claims extracted")
    return _status(ok)


def run_retrieval_test() -> str:
    """Test FAISS retrieval if index exists."""
    from config import Config
    from core.evidence_retriever import EvidenceRetriever

    config = Config()
    if not Path(config.retrieval.index_path).exists():
        print(f"  Retrieval test: {WARN} — skipped (no index)")
        return WARN

    retriever = EvidenceRetriever(config)
    results = retriever.retrieve("Albert Einstein Nobel Prize Physics")
    ok = len(results) > 0
    print(f"  Retrieval test: {_status(ok)} — {len(results)} evidence passages found")
    return _status(ok)


def main() -> None:
    print("=" * 60)
    print("VeriFactAI Smoke Test")
    print("=" * 60)

    print("\n[1/3] Prerequisites:")
    results = [
        check_python_version(),
        check_spacy_model(),
        check_ollama(),
        check_faiss_index(),
        check_embedding_model(),
        check_nli_model(),
    ]

    print("\n[2/3] Component tests:")
    results.append(run_decomposition_test())
    results.append(run_retrieval_test())

    print("\n[3/3] Summary:")
    failures = results.count(FAIL)
    warnings = results.count(WARN)
    passes = results.count(PASS)
    print(f"  {passes} passed, {warnings} warnings, {failures} failures")

    if failures == 0:
        print("\n  VERDICT: READY for demo")
        sys.exit(0)
    else:
        print("\n  VERDICT: FIX failures above before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
