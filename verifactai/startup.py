#!/usr/bin/env python3
"""
VeriFACT AI — Production Startup Script.

Ensures FAISS index exists, then starts the API server.
If no index found, builds one (2500 articles, ~12 min on free CPU).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.runtime_safety import apply_runtime_safety_defaults

apply_runtime_safety_defaults()


def main() -> None:
    from config import cfg

    index_path = Path(cfg.retrieval.index_path)
    meta_path = Path(cfg.retrieval.metadata_path)

    if not index_path.exists() or not meta_path.exists():
        print("=" * 50)
        print("  No FAISS index found. Building knowledge base...")
        print("  Sources: Wikipedia + SQuAD + PubMed + FEVER")
        print("  This takes ~15-20 minutes on first deploy.")
        print("=" * 50)

        max_articles = int(os.environ.get("VERIFACT_INDEX_SIZE", "2500"))

        from data.build_index import (
            build_and_save,
            load_fever,
            load_pubmed,
            load_squad,
            load_wikipedia,
        )

        # Build multi-source knowledge base
        chunks = load_wikipedia(max_articles)

        # SQuAD: high-quality Wikipedia reading comprehension passages
        try:
            chunks.extend(load_squad())
        except Exception as e:
            print(f"  SQuAD failed: {e}, continuing...")

        # PubMed: medical/scientific knowledge
        try:
            chunks.extend(load_pubmed())
        except Exception as e:
            print(f"  PubMed failed: {e}, continuing...")

        # FEVER: fact verification claims with labels
        try:
            chunks.extend(load_fever())
        except Exception as e:
            print(f"  FEVER failed: {e}, continuing...")

        # Skip Natural Questions on HF free tier (too slow)
        os.environ["VERIFACT_SKIP_NQ"] = "1"

        build_and_save(chunks, cfg.retrieval.index_path, cfg.retrieval.metadata_path)
        print(f"Index ready: {index_path}")
    else:
        size_mb = index_path.stat().st_size / (1024 * 1024)
        print(f"FAISS index found: {size_mb:.0f} MB")

    # Start the API server
    print("Starting API server...")
    from overlay_server import run

    run()


if __name__ == "__main__":
    main()
