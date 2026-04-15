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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation rules."""
    import re

    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk_text(
    text: str,
    title: str,
    source: str,
    url: str = "",
    chunk_size: int = 200,
    overlap: int = 50,
    min_words: int = 30,
) -> list[dict]:
    """Semantic paragraph-aware chunking with sentence-boundary splitting.

    Strategy:
    1. Split on paragraph breaks (double newline).
    2. Merge short paragraphs (< 50 words) with the next one.
    3. Split long paragraphs (> 300 words) at sentence boundaries.
    4. Target chunk size: 100-250 words.
    5. Overlap: last 2 sentences of previous chunk prepended to next.
    """
    # ── Step 1: split into paragraphs ──
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # ── Step 2: merge short paragraphs (< 50 words) ──
    merged: list[str] = []
    buffer = ""
    for para in raw_paragraphs:
        if buffer:
            buffer = buffer + " " + para
        else:
            buffer = para
        if len(buffer.split()) >= 50:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    # ── Step 3: split long paragraphs at sentence boundaries (target 100-250 words) ──
    segments: list[str] = []
    for para in merged:
        word_count = len(para.split())
        if word_count <= 300:
            segments.append(para)
        else:
            # Split at sentence boundaries into ~100-250 word chunks
            sentences = _split_sentences(para)
            current: list[str] = []
            current_wc = 0
            for sent in sentences:
                sent_wc = len(sent.split())
                if current_wc + sent_wc > 250 and current_wc >= 100:
                    segments.append(" ".join(current))
                    current = [sent]
                    current_wc = sent_wc
                else:
                    current.append(sent)
                    current_wc += sent_wc
            if current:
                # If leftover is too small, merge with previous segment
                if current_wc < min_words and segments:
                    segments[-1] = segments[-1] + " " + " ".join(current)
                else:
                    segments.append(" ".join(current))

    # ── Step 4: build chunks with 2-sentence overlap ──
    chunks: list[dict] = []
    prev_last_two_sentences: list[str] = []
    for seg in segments:
        seg_words = seg.split()
        if len(seg_words) < min_words:
            continue
        # Prepend overlap from previous chunk
        if prev_last_two_sentences:
            overlap_text = " ".join(prev_last_two_sentences)
            chunk_text_final = overlap_text + " " + seg
        else:
            chunk_text_final = seg
        # Trim if overlap pushed it too far beyond 250 words
        final_words = chunk_text_final.split()
        if len(final_words) > 300:
            chunk_text_final = " ".join(final_words[:300])
        chunks.append(
            {
                "text": chunk_text_final,
                "source": source,
                "title": title,
                "url": url,
            }
        )
        # Track last 2 sentences for next overlap
        sentences = _split_sentences(seg)
        prev_last_two_sentences = sentences[-2:] if len(sentences) >= 2 else sentences

    # Fallback: if semantic chunking produced nothing (e.g. no paragraph breaks),
    # use simple word-level chunking
    if not chunks and len(text.split()) >= min_words:
        words = text.split()
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < min_words:
                continue
            chunks.append(
                {
                    "text": " ".join(chunk_words),
                    "source": source,
                    "title": title,
                    "url": url,
                }
            )
    return chunks


def _ensure_topic_diversity(wiki_dataset, existing_titles: set[str], max_extra: int = 500) -> list[dict]:
    """Fetch specific articles for underrepresented topics if missing from random sample.

    Checks coverage across key topic categories and fetches targeted articles
    to fill gaps. Returns a list of article dicts (title, text) to add.
    """
    # Key topics that should be represented for good fact-checking coverage
    topic_seeds: dict[str, list[str]] = {
        "countries": [
            "United States", "China", "India", "Brazil", "Russia", "Japan",
            "Germany", "France", "United Kingdom", "Australia", "Canada",
            "Mexico", "South Africa", "Egypt", "Nigeria", "Indonesia",
            "South Korea", "Italy", "Spain", "Argentina",
        ],
        "capitals": [
            "Washington, D.C.", "Beijing", "New Delhi", "Tokyo", "Berlin",
            "Paris", "London", "Canberra", "Ottawa", "Moscow",
            "Brasília", "Cairo", "Rome", "Madrid", "Buenos Aires",
        ],
        "famous_people": [
            "Albert Einstein", "Isaac Newton", "Marie Curie", "Charles Darwin",
            "Mahatma Gandhi", "Nelson Mandela", "Martin Luther King Jr.",
            "Leonardo da Vinci", "William Shakespeare", "Cleopatra",
            "Alexander the Great", "Napoleon", "Abraham Lincoln",
            "Nikola Tesla", "Ada Lovelace",
        ],
        "scientific_facts": [
            "Speed of light", "DNA", "Evolution", "Gravity", "Atom",
            "Solar System", "Photosynthesis", "Vaccine", "Periodic table",
            "Climate change", "Cell (biology)", "Virus", "Antibiotic",
            "Quantum mechanics", "Theory of relativity",
        ],
        "historical_events": [
            "World War I", "World War II", "French Revolution",
            "American Revolution", "Industrial Revolution", "Cold War",
            "Moon landing", "Fall of the Berlin Wall", "Renaissance",
            "Ancient Rome", "Ancient Egypt", "Ancient Greece",
            "Printing press", "Declaration of Independence",
        ],
    }

    existing_lower = {t.lower() for t in existing_titles}
    missing: list[str] = []
    for _category, titles in topic_seeds.items():
        for title in titles:
            if title.lower() not in existing_lower:
                missing.append(title)

    if not missing:
        print("       Topic diversity: all key topics already covered")
        return []

    print(f"       Topic diversity: {len(missing)} key articles missing, searching dataset …")

    # Build a lookup from the full dataset by title
    title_to_article: dict[str, dict] = {}
    for article in wiki_dataset:
        t = article["title"]
        if t.lower() in {m.lower() for m in missing}:
            title_to_article[t.lower()] = {"title": article["title"], "text": article["text"]}

    extra_articles = list(title_to_article.values())[:max_extra]
    found = len(extra_articles)
    print(f"       Topic diversity: found {found}/{len(missing)} missing key articles in dataset")
    return extra_articles


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
                repo,
                config,
                split="train",
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
        wiki_subset = wiki.select(range(min(max_articles, len(wiki))))
    else:
        wiki_subset = wiki
    print(f"       {len(wiki_subset):,} articles loaded")

    # Track titles for diversity check
    indexed_titles: set[str] = set()
    all_chunks: list[dict] = []
    for article in tqdm(wiki_subset, desc="Chunking Wikipedia"):
        title = article["title"]
        text = article["text"]
        if not text or len(text.split()) < 30:
            continue
        indexed_titles.add(title)
        url = f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"
        all_chunks.extend(
            chunk_text(
                text,
                title,
                "wikipedia",
                url,
                cfg.retrieval.chunk_size,
                cfg.retrieval.chunk_overlap,
            )
        )

    # ── Topic diversity: fill gaps for key articles ──
    extra_articles = _ensure_topic_diversity(wiki, indexed_titles)
    for article in extra_articles:
        title = article["title"]
        text = article["text"]
        if not text or len(text.split()) < 30:
            continue
        url = f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"
        all_chunks.extend(
            chunk_text(
                text,
                title,
                "wikipedia",
                url,
                cfg.retrieval.chunk_size,
                cfg.retrieval.chunk_overlap,
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
                context_text,
                title,
                "pubmed",
                url,
                cfg.retrieval.chunk_size,
                cfg.retrieval.chunk_overlap,
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
        "--max-articles",
        type=int,
        default=None,
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
