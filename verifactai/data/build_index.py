#!/usr/bin/env python3
"""
Knowledge Corpus Builder for VeriFactAI.

Downloads Wikipedia Simple English + PubMed QA → chunks → embeds →
builds FAISS IndexFlatIP → saves index + metadata.

Usage:
    python data/build_index.py                   # full build
    python data/build_index.py --wiki-only       # Wikipedia only (faster)
    python data/build_index.py --max-articles 50000  # subset for quick dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from utils.runtime_safety import apply_runtime_safety_defaults, configure_faiss_threads

apply_runtime_safety_defaults()

import faiss
import numpy as np
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg  # noqa: E402

configure_faiss_threads()


def chunk_text(
    text: str,
    title: str,
    source: str,
    url: str = "",
    chunk_size: int = 200,
    overlap: int = 50,
    min_words: int = 30,
) -> list[dict]:
    """Split text into overlapping word-level chunks with metadata."""
    words = text.split()
    chunks: list[dict] = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < min_words:
            continue
        chunks.append({
            "text": " ".join(chunk_words),
            "source": source,
            "title": title,
            "url": url,
        })
    return chunks


def load_wikipedia(max_articles: int | None = None) -> list[dict]:
    """Download and chunk Wikipedia Simple English."""
    from datasets import load_dataset

    print("[1/4] Downloading Wikipedia Simple English …")
    # Try multiple dataset repos + configs (HF API changes across versions)
    wiki = None
    attempts = [
        ("wikimedia/wikipedia", "20231101.simple"),
        ("wikimedia/wikipedia", "20220301.simple"),
        ("legacy-datasets/wikipedia", "20231101.simple"),
        ("legacy-datasets/wikipedia", "20220301.simple"),
    ]
    for repo, config in attempts:
        try:
            print(f"       Trying {repo} / {config} …")
            wiki = load_dataset(
                repo, config, split="train",
                trust_remote_code=True,
                storage_options={"client_kwargs": {"timeout": 300}},
            )
            print(f"       Loaded from {repo} ({config})")
            break
        except Exception as exc:
            print(f"       Failed: {exc.__class__.__name__}")
            continue
    if wiki is None:
        raise RuntimeError(
            "Could not load Wikipedia dataset from any source. Try:\n"
            "  1. Set HF_TOKEN env var for authenticated access\n"
            "  2. pip install apache-beam mwparserfromhell\n"
            "  3. Check network connection and retry"
        )
    if max_articles:
        wiki = wiki.select(range(min(max_articles, len(wiki))))
    print(f"       {len(wiki):,} articles loaded")

    all_chunks: list[dict] = []
    for article in tqdm(wiki, desc="Chunking Wikipedia"):
        title = article["title"]
        text = article["text"]
        if not text or len(text.split()) < 30:
            continue
        url = f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"
        all_chunks.extend(
            chunk_text(
                text, title, "wikipedia", url,
                cfg.retrieval.chunk_size, cfg.retrieval.chunk_overlap,
            )
        )
    print(f"       {len(all_chunks):,} Wikipedia chunks created")
    return all_chunks


def load_pubmed() -> list[dict]:
    """Download and chunk PubMed QA abstracts."""
    from datasets import load_dataset

    print("[2/4] Downloading PubMed QA …")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception:
        # Fallback: try alternate name
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    print(f"       {len(ds):,} entries loaded")

    all_chunks: list[dict] = []
    for item in tqdm(ds, desc="Chunking PubMed"):
        contexts = item.get("context", {}).get("contexts", [])
        context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
        if not context_text or len(context_text.split()) < 20:
            continue
        pubid = item.get("pubid", "")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pubid}" if pubid else ""
        title = item.get("question", "PubMed Abstract")
        all_chunks.extend(
            chunk_text(
                context_text, title, "pubmed", url,
                cfg.retrieval.chunk_size, cfg.retrieval.chunk_overlap,
            )
        )
    print(f"       {len(all_chunks):,} PubMed chunks created")
    return all_chunks


def build_and_save(
    all_chunks: list[dict],
    index_path: str,
    metadata_path: str,
) -> None:
    """Embed chunks, build FAISS index, persist to disk."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Save metadata
    print("[3/4] Saving metadata …")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"       Metadata written to {metadata_path}")

    # Embed
    print(f"[4/4] Generating embeddings with {cfg.embedding.model_name} …")
    encoder = SentenceTransformer(cfg.embedding.model_name)
    texts = [c["text"] for c in all_chunks]
    embeddings = encoder.encode(
        texts,
        batch_size=cfg.embedding.batch_size,
        show_progress_bar=True,
        normalize_embeddings=cfg.embedding.normalize,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"       Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via normalised inner product
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"       FAISS index saved to {index_path}  ({index.ntotal:,} vectors)")
    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VeriFactAI knowledge index")
    parser.add_argument("--wiki-only", action="store_true", help="Skip PubMed")
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="Limit Wikipedia articles (for quick dev builds)",
    )
    args = parser.parse_args()

    all_chunks = load_wikipedia(args.max_articles)
    if not args.wiki_only:
        all_chunks.extend(load_pubmed())

    print(f"\nTotal corpus: {len(all_chunks):,} chunks")
    build_and_save(
        all_chunks,
        cfg.retrieval.index_path,
        cfg.retrieval.metadata_path,
    )


if __name__ == "__main__":
    main()
