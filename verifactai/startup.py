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
        print("  No FAISS index found. Building small index...")
        print("  This takes ~2-3 minutes on first deploy.")
        print("=" * 50)

        # Build index — 2500 articles balances coverage vs free-tier build time (~12 min)
        max_articles = int(os.environ.get("VERIFACT_INDEX_SIZE", "2500"))

        from data.build_index import build_and_save, load_wikipedia

        chunks = load_wikipedia(max_articles)
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
