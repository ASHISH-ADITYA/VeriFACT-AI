"""
Evidence Retriever — Stage 2 of the VeriFactAI pipeline.

Hybrid retrieval over:
1. Pre-built FAISS index (local Wikipedia + FEVER chunks)
2. Live Wikipedia API search (6.8M articles, fallback when local evidence is weak)
3. BM25 sparse keyword matching

Returns top-k evidence passages per claim with metadata.
Supports both single-claim and batch retrieval for throughput.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from utils.runtime_safety import apply_runtime_safety_defaults, configure_faiss_threads

apply_runtime_safety_defaults()

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.helpers import logger, timed

if TYPE_CHECKING:
    from config import Config


@dataclass
class Evidence:
    """A single evidence passage retrieved from the knowledge corpus."""

    text: str
    source: str  # "wikipedia" | "pubmed"
    title: str
    url: str
    similarity: float
    chunk_id: int
    # Entity verification flags (set by retriever)
    entity_verified: bool | None = field(default=None, repr=False)
    unverified_entities: list[str] = field(default_factory=list, repr=False)


class EvidenceRetriever:
    """Hybrid retriever: dense (FAISS) + sparse (BM25) + cross-encoder reranker."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._faiss_lock = threading.Lock()
        self._use_numpy_search = os.environ.get("VERIFACTAI_USE_NUMPY_SEARCH", "0") == "1"
        self._embedding_matrix: np.ndarray | None = None
        self._bm25 = None
        self._reranker = None
        configure_faiss_threads()
        rc = config.retrieval
        ec = config.embedding

        logger.info(f"Loading embedding model: {ec.model_name}")
        self.encoder = SentenceTransformer(ec.model_name)

        index_path = Path(rc.index_path)
        meta_path = Path(rc.metadata_path)

        if not index_path.exists():
            logger.warning(f"FAISS index not found at {index_path}. Run data/build_index.py first.")
            self.index: faiss.Index | None = None
            self.metadata: list[dict] = []
            return

        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Index loaded — {self.index.ntotal:,} vectors")

        if self._use_numpy_search:
            logger.warning("VERIFACTAI_USE_NUMPY_SEARCH=1: using NumPy retrieval fallback")
            self._embedding_matrix = self.index.reconstruct_n(0, self.index.ntotal).astype(
                np.float32
            )
            if self.config.embedding.normalize:
                norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
                self._embedding_matrix = self._embedding_matrix / np.clip(norms, 1e-12, None)

        logger.info(f"Loading metadata from {meta_path}")
        self.metadata = self._load_metadata(meta_path)

        # ── BM25 sparse index ─────────────────────────────────
        self._init_bm25()

        # ── Cross-encoder reranker ────────────────────────────
        self._init_reranker()

    def _init_bm25(self) -> None:
        """Build BM25 index from metadata texts (skip if corpus too large for interactive use)."""
        if os.environ.get("VERIFACT_DISABLE_BM25", "0") == "1":
            logger.info("BM25 disabled via VERIFACT_DISABLE_BM25")
            return
        # BM25 is always enabled — hybrid retrieval improves recall at all corpus sizes
        try:
            from rank_bm25 import BM25Okapi

            corpus = [m.get("text", "") for m in self.metadata]
            tokenized = [doc.lower().split() for doc in corpus]
            self._bm25 = BM25Okapi(tokenized)
            logger.info(f"BM25 index built ({len(corpus):,} documents)")
        except ImportError:
            logger.warning("rank-bm25 not installed — using dense retrieval only")
        except Exception as exc:
            logger.warning(f"BM25 init failed ({exc}) — using dense retrieval only")

    def _init_reranker(self) -> None:
        """Load cross-encoder reranker for evidence quality improvement."""
        if os.environ.get("VERIFACT_DISABLE_RERANKER", "0") == "1":
            logger.info("Reranker disabled via VERIFACT_DISABLE_RERANKER")
            return
        try:
            from sentence_transformers import CrossEncoder

            model_name = os.environ.get(
                "VERIFACT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            self._reranker = CrossEncoder(model_name)
            logger.info(f"Cross-encoder reranker loaded: {model_name}")
        except Exception as exc:
            logger.warning(f"Reranker init failed ({exc}) — skipping reranking")

    # ------------------------------------------------------------------
    # Contradiction-aware query expansion
    # ------------------------------------------------------------------
    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "of", "in", "to",
        "for", "with", "on", "at", "from", "by", "about", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "that", "this", "these", "those", "it", "its", "not", "no", "nor",
        "but", "or", "and", "if", "than", "too", "very", "just", "so",
    })

    def _expand_queries(self, claim: str) -> list[str]:
        """Generate expanded queries including a negation-oriented query.

        For a claim like "The Great Wall is in South America", produces:
        1. The original claim (always first)
        2. A negation query built from key content words, stripping
           potentially false location/attribute words and adding likely
           correct ones (e.g. "Great Wall China location Asia")

        This helps retrieve contradicting evidence that the original
        query embedding would miss.
        """
        queries = [claim]

        # Extract content words (nouns, proper nouns, key terms)
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", claim)
        content_words = [
            w for w in words if w.lower() not in self._STOPWORDS and len(w) > 1
        ]

        if len(content_words) < 2:
            return queries

        # Build negation query: keep entity words, add "location origin facts"
        # to nudge retrieval toward factual context rather than the claim's assertion
        negation_query = " ".join(content_words) + " facts origin location history"
        queries.append(negation_query)

        return queries

    # ------------------------------------------------------------------
    # Entity verification: check if Wikipedia has an exact-title article
    # ------------------------------------------------------------------
    _COMMON_PHRASES = frozenset({
        "United States", "New York", "South America", "North America",
        "European Union", "Middle East", "World War", "Nobel Prize",
        "Prime Minister", "President Biden", "President Obama",
        "High School", "Real Estate", "Social Media", "Machine Learning",
        "Deep Learning", "Artificial Intelligence", "Computer Science",
        "Natural Language", "Climate Change", "Global Warming",
        "Big Bang", "Black Hole", "Dark Matter", "Dark Energy",
        "Solar System", "Milky Way", "General Theory", "Special Theory",
    })

    @staticmethod
    def _verify_entity_exists(entity_name: str) -> bool:
        """Check if Wikipedia has an article with exactly this title.

        Uses the MediaWiki query API with exact title lookup.
        Returns True only if an article exists (no 'missing' key).
        """
        url = (
            f"https://en.wikipedia.org/w/api.php?action=query"
            f"&titles={urllib.parse.quote(entity_name)}"
            f"&format=json&utf8=1"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VeriFACT-AI/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for _page_id, page_info in pages.items():
                return "missing" not in page_info
            return False
        except Exception:
            # On network errors, return None-ish — don't block the pipeline
            return True  # fail-open: assume it exists if we can't check

    @staticmethod
    def _extract_specific_entities(claim: str) -> list[str]:
        """Extract specific named entities from a claim that could be fabricated.

        Looks for:
        - Quoted terms: "Haldiram Novel Theory"
        - Capitalized multi-word phrases: Haldiram Novel Theory, Progressimic Algorithm
        - Technical-sounding compound terms
        """
        entities: list[str] = []

        # 1. Quoted terms
        quoted = re.findall(r'"([^"]+)"', claim)
        entities.extend(quoted)

        # 2. Capitalized multi-word phrases (2+ consecutive capitalized words)
        # Match sequences like "Haldiram Novel Theory", "Progressimic Algorithm"
        cap_phrases = re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', claim)
        for phrase in cap_phrases:
            words = phrase.split()
            # Skip very common phrases that are not entity names
            if len(words) >= 2 and phrase not in EvidenceRetriever._COMMON_PHRASES:
                entities.append(phrase)

        # 3. Single capitalized word + technical suffix (e.g., "Progressimic Algorithm")
        # Already covered by multi-word above

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique.append(e)

        return unique

    @staticmethod
    def _check_title_relevance(claim: str, evidence_title: str) -> float:
        """Check if a Wikipedia article title is actually relevant to the claim.

        Returns a penalty multiplier (0.0 to 1.0).
        1.0 = fully relevant, keep original similarity
        Low value = irrelevant, should reduce similarity

        The key check: if the claim mentions a specific concept like
        "Haldiram Novel Theory", the Wikipedia article "Haldiram's" is NOT
        about that theory — it's about a snack company.
        """
        # Extract key terms from the claim (excluding very common words)
        stop = {
            "the", "a", "an", "is", "was", "are", "were", "in", "on", "at",
            "of", "to", "and", "or", "for", "by", "it", "its", "that", "this",
            "with", "from", "as", "be", "has", "had", "not", "said", "says",
            "about", "also", "been", "but", "can", "could", "did", "do",
            "does", "each", "even", "get", "got", "have", "he", "her", "here",
            "him", "his", "how", "into", "just", "know", "like", "make",
            "many", "may", "more", "most", "much", "my", "new", "no", "now",
            "only", "our", "out", "over", "own", "part", "per", "put",
            "she", "so", "some", "still", "such", "take", "than", "their",
            "them", "then", "there", "they", "too", "up", "us", "use",
            "used", "very", "want", "way", "we", "well", "what", "when",
            "where", "which", "while", "who", "why", "will", "would", "you",
        }

        claim_words = {w.lower() for w in re.findall(r'[A-Za-z]+', claim) if len(w) > 2}
        claim_content = claim_words - stop

        title_words = {w.lower() for w in re.findall(r'[A-Za-z]+', evidence_title) if len(w) > 2}

        if not claim_content or not title_words:
            return 1.0

        # How many claim content words appear in the title?
        overlap = claim_content & title_words
        overlap_ratio = len(overlap) / max(len(claim_content), 1)

        # If the title covers less than 40% of the claim's content words,
        # it's probably about something else
        if overlap_ratio < 0.3:
            return 0.2  # Heavily penalize
        if overlap_ratio < 0.5:
            return 0.5  # Moderate penalty

        return 1.0  # Relevant enough

    def _verify_claim_entities(self, claim_text: str) -> tuple[bool, list[str]]:
        """Verify all specific entities in a claim exist on Wikipedia.

        Returns (all_verified, list_of_unverified_entity_names).
        """
        entities = self._extract_specific_entities(claim_text)
        if not entities:
            return True, []

        unverified: list[str] = []
        for entity in entities:
            if not self._verify_entity_exists(entity):
                unverified.append(entity)

        return len(unverified) == 0, unverified

    # ------------------------------------------------------------------
    @timed
    def retrieve(self, claim_text: str, top_k: int | None = None) -> list[Evidence]:
        """Hybrid retrieve with contradiction-aware query expansion: dense + BM25 + rerank."""
        if self.index is None:
            return []
        k = top_k or self.config.retrieval.top_k
        # Fetch moderate candidates — too many makes reranking slow on CPU
        fetch_k = max(k * 3, 10)

        # Expand queries for contradiction-aware retrieval
        queries = self._expand_queries(claim_text)

        # Collect candidates from all expanded queries
        seen_chunk_ids: set[int] = set()
        all_candidates: list[Evidence] = []
        for query in queries:
            candidates = self._hybrid_fetch(query, fetch_k)
            for ev in candidates:
                if ev.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(ev.chunk_id)
                    all_candidates.append(ev)

        reranked = self._rerank(claim_text, all_candidates, k)

        # ── Entity verification: check if specific entities exist on Wikipedia ──
        all_verified, unverified = self._verify_claim_entities(claim_text)

        # ALWAYS query live Wikipedia API (6.8M articles) for every claim
        # This ensures maximum evidence coverage — not just a fallback
        try:
            wiki_results = self._wiki_api_search(claim_text, max_results=3)
            if wiki_results:
                logger.info(f"Wiki API: {len(wiki_results)} live results for: {claim_text[:50]}")
                # ── Fix 3: Apply title relevance scoring to Wiki results ──
                for i, wev in enumerate(wiki_results):
                    relevance = self._check_title_relevance(claim_text, wev.title)
                    if relevance < 1.0:
                        wiki_results[i] = Evidence(
                            text=wev.text,
                            source=wev.source,
                            title=wev.title,
                            url=wev.url,
                            similarity=round(wev.similarity * relevance, 3),
                            chunk_id=wev.chunk_id,
                            entity_verified=wev.entity_verified,
                            unverified_entities=wev.unverified_entities,
                        )
                # Merge: keep local results + add wiki results, deduplicate by title
                local_titles = {ev.title.lower() for ev in reranked}
                for wev in wiki_results:
                    if wev.title.lower() not in local_titles:
                        reranked.append(wev)
                # Sort by similarity descending, cap at top_k
                reranked.sort(key=lambda ev: ev.similarity, reverse=True)
                reranked = reranked[:max(k, 5)]  # keep at least 5 for better NLI
            else:
                logger.info(f"Wiki API: 0 results — possible fabrication: {claim_text[:50]}")
        except Exception:
            pass  # fail silently — local results are still available

        # ── Stamp entity verification results on all evidence ──
        for ev in reranked:
            ev.entity_verified = all_verified
            ev.unverified_entities = unverified

        return reranked

    @timed
    def batch_retrieve(self, claims: list[str], top_k: int | None = None) -> list[list[Evidence]]:
        """True batch hybrid retrieval: dense + BM25 + RRF + batch rerank."""
        if self.index is None or not claims:
            return [[] for _ in claims]

        k = top_k or self.config.retrieval.top_k
        fetch_k = max(k * 3, 10)
        n = len(claims)

        # ── Step 1: Batch dense retrieval (encode + FAISS search all at once) ──
        embeddings = self.encoder.encode(
            claims,
            normalize_embeddings=self.config.embedding.normalize,
            batch_size=len(claims),
            show_progress_bar=False,
        )
        query_batch = np.array(embeddings, dtype=np.float32)

        if self._use_numpy_search and self._embedding_matrix is not None:
            scores_batch, indices_batch = self._numpy_search(query_batch, fetch_k)
        else:
            with self._faiss_lock:
                scores_batch, indices_batch = self.index.search(query_batch, fetch_k)

        dense_per_claim = [
            self._build_evidence(scores_batch[i], indices_batch[i])
            for i in range(n)
        ]

        # ── Step 2: BM25 sparse retrieval (per-claim, BM25 doesn't batch) ──
        bm25_per_claim: list[list[Evidence]] = []
        for claim in claims:
            bm25_per_claim.append(
                self._bm25_fetch(claim, fetch_k) if self._bm25 else []
            )

        # ── Step 3: Reciprocal Rank Fusion per claim ──
        candidates_per_claim: list[list[Evidence]] = []
        for i in range(n):
            dense_results = dense_per_claim[i]
            bm25_results = bm25_per_claim[i]

            if not bm25_results:
                candidates_per_claim.append(dense_results)
                continue

            rrf_k = 60
            scored: dict[int, float] = {}
            evidence_map: dict[int, Evidence] = {}

            for rank, ev in enumerate(dense_results):
                scored[ev.chunk_id] = scored.get(ev.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
                evidence_map[ev.chunk_id] = ev

            for rank, ev in enumerate(bm25_results):
                scored[ev.chunk_id] = scored.get(ev.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
                if ev.chunk_id not in evidence_map:
                    evidence_map[ev.chunk_id] = ev

            sorted_ids = sorted(scored, key=lambda cid: scored[cid], reverse=True)
            candidates_per_claim.append(
                [evidence_map[cid] for cid in sorted_ids[:fetch_k]]
            )

        # ── Step 4: Batch rerank across ALL claims in one predict() call ──
        if self._reranker is None:
            return [cands[:k] for cands in candidates_per_claim]

        # Build a single flat list of (query, passage) pairs with a mapping back
        all_pairs: list[tuple[str, str]] = []
        pair_map: list[tuple[int, int]] = []  # (claim_idx, candidate_idx)
        needs_rerank: list[bool] = []

        for i in range(n):
            cands = candidates_per_claim[i]
            if len(cands) <= k:
                needs_rerank.append(False)
            else:
                needs_rerank.append(True)
                for j, ev in enumerate(cands):
                    all_pairs.append((claims[i], ev.text))
                    pair_map.append((i, j))

        if not all_pairs:
            return [cands[:k] for cands in candidates_per_claim]

        try:
            rerank_scores = self._reranker.predict(all_pairs, show_progress_bar=False)
        except Exception as exc:
            logger.warning(f"Batch reranking failed ({exc}) — returning un-reranked results")
            return [cands[:k] for cands in candidates_per_claim]

        # Distribute scores back per claim
        score_lookup: dict[tuple[int, int], float] = {}
        for idx, (ci, cj) in enumerate(pair_map):
            score_lookup[(ci, cj)] = float(rerank_scores[idx])

        results: list[list[Evidence]] = []
        for i in range(n):
            cands = candidates_per_claim[i]
            if not needs_rerank[i]:
                results.append(cands[:k])
                continue

            scored_cands = [
                (ev, score_lookup[(i, j)]) for j, ev in enumerate(cands)
            ]
            scored_cands.sort(key=lambda x: x[1], reverse=True)
            reranked = [
                Evidence(
                    text=ev.text,
                    source=ev.source,
                    title=ev.title,
                    url=ev.url,
                    similarity=float(1 / (1 + np.exp(-score))),
                    chunk_id=ev.chunk_id,
                )
                for ev, score in scored_cands[:k]
            ]
            # ── Entity verification for this claim ──
            all_verified, unverified = self._verify_claim_entities(claims[i])

            # ALWAYS query Wikipedia API for every claim
            try:
                wiki_ev = self._wiki_api_search(claims[i], max_results=2)
                if wiki_ev:
                    # ── Fix 3: Apply title relevance scoring ──
                    for wi, wev in enumerate(wiki_ev):
                        relevance = self._check_title_relevance(claims[i], wev.title)
                        if relevance < 1.0:
                            wiki_ev[wi] = Evidence(
                                text=wev.text,
                                source=wev.source,
                                title=wev.title,
                                url=wev.url,
                                similarity=round(wev.similarity * relevance, 3),
                                chunk_id=wev.chunk_id,
                            )
                    local_titles = {ev.title.lower() for ev in reranked}
                    for wev in wiki_ev:
                        if wev.title.lower() not in local_titles:
                            reranked.append(wev)
                    reranked.sort(key=lambda ev: ev.similarity, reverse=True)
                    reranked = reranked[:max(k, 5)]
            except Exception:
                pass

            # ── Stamp entity verification results on all evidence ──
            for ev in reranked:
                ev.entity_verified = all_verified
                ev.unverified_entities = unverified

            results.append(reranked)

        return results

    # ------------------------------------------------------------------
    # Fast retrieval: FAISS-only, no BM25 / reranker (for /analyze/fast)
    # ------------------------------------------------------------------
    def fast_retrieve(
        self, claims: list[str], top_k: int | None = None
    ) -> list[list[Evidence]]:
        """Batch FAISS-only retrieval — no BM25, no reranker.

        Designed for the extension fast-path where latency matters more
        than marginal recall gains.  Returns top-k evidence per claim
        (default: 3).
        """
        if self.index is None or not claims:
            return [[] for _ in claims]

        k = top_k or min(self.config.retrieval.top_k, 3)

        embeddings = self.encoder.encode(
            claims,
            normalize_embeddings=self.config.embedding.normalize,
            batch_size=len(claims),
            show_progress_bar=False,
        )
        query_batch = np.array(embeddings, dtype=np.float32)

        if self._use_numpy_search and self._embedding_matrix is not None:
            scores_batch, indices_batch = self._numpy_search(query_batch, k)
        else:
            with self._faiss_lock:
                scores_batch, indices_batch = self.index.search(query_batch, k)

        results = [
            self._build_evidence(scores_batch[i], indices_batch[i])
            for i in range(len(claims))
        ]

        # Entity verification + Wikipedia API for each claim (critical for fabrication detection)
        for i, claim_text in enumerate(claims):
            # Entity verification: does Wikipedia have an article with this exact name?
            all_verified, unverified = self._verify_claim_entities(claim_text)

            # Wikipedia API search (ALWAYS, for every claim)
            try:
                wiki_ev = self._wiki_api_search(claim_text, max_results=2)
                if wiki_ev:
                    # Apply title relevance check
                    for wev in wiki_ev:
                        relevance = self._check_title_relevance(claim_text, wev.title)
                        wev.similarity = round(wev.similarity * relevance, 3)
                    local_titles = {ev.title.lower() for ev in results[i]}
                    for wev in wiki_ev:
                        if wev.title.lower() not in local_titles:
                            results[i].append(wev)
                    results[i].sort(key=lambda ev: ev.similarity, reverse=True)
                    results[i] = results[i][:max(k, 5)]
            except Exception:
                pass

            # Stamp entity verification flags on all evidence for this claim
            for ev in results[i]:
                ev.entity_verified = all_verified
                ev.unverified_entities = unverified

        return results

    # ------------------------------------------------------------------
    # Hybrid fetch: dense FAISS + sparse BM25 with reciprocal rank fusion
    # ------------------------------------------------------------------
    def _hybrid_fetch(self, query: str, fetch_k: int) -> list[Evidence]:
        """Combine dense and sparse retrieval with reciprocal rank fusion."""
        # Dense retrieval (FAISS)
        dense_results = self._dense_fetch(query, fetch_k)

        # Sparse retrieval (BM25)
        bm25_results = self._bm25_fetch(query, fetch_k) if self._bm25 else []

        if not bm25_results:
            return dense_results

        # Reciprocal Rank Fusion (RRF)
        rrf_k = 60  # standard RRF constant
        scored: dict[int, float] = {}  # chunk_id → fused score
        evidence_map: dict[int, Evidence] = {}

        for rank, ev in enumerate(dense_results):
            scored[ev.chunk_id] = scored.get(ev.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            evidence_map[ev.chunk_id] = ev

        for rank, ev in enumerate(bm25_results):
            scored[ev.chunk_id] = scored.get(ev.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            if ev.chunk_id not in evidence_map:
                evidence_map[ev.chunk_id] = ev

        # Sort by fused score descending
        sorted_ids = sorted(scored, key=lambda cid: scored[cid], reverse=True)
        return [evidence_map[cid] for cid in sorted_ids[:fetch_k]]

    def _dense_fetch(self, query: str, k: int) -> list[Evidence]:
        """Pure FAISS dense retrieval."""
        embedding = self.encoder.encode([query], normalize_embeddings=self.config.embedding.normalize)
        query_vec = np.array(embedding, dtype=np.float32)
        if self._use_numpy_search and self._embedding_matrix is not None:
            scores, indices = self._numpy_search(query_vec, k)
        else:
            with self._faiss_lock:
                scores, indices = self.index.search(query_vec, k)
        return self._build_evidence(scores[0], indices[0])

    def _bm25_fetch(self, query: str, k: int) -> list[Evidence]:
        """BM25 sparse retrieval."""
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        threshold = self.config.retrieval.relevance_threshold
        results = []
        for idx in top_indices:
            idx = int(idx)
            score = float(scores[idx])
            if score <= 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            # Normalize BM25 score to 0-1 range (approximate)
            norm_score = min(score / 30.0, 1.0)
            if norm_score < threshold:
                continue
            results.append(Evidence(
                text=meta["text"],
                source=meta.get("source", "unknown"),
                title=meta.get("title", ""),
                url=meta.get("url", ""),
                similarity=norm_score,
                chunk_id=idx,
            ))
        return results

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------
    def _rerank(self, query: str, candidates: list[Evidence], top_k: int) -> list[Evidence]:
        """Rerank candidates using cross-encoder for better evidence quality."""
        if not candidates:
            return []
        if self._reranker is None or len(candidates) <= top_k:
            return candidates[:top_k]

        # Cross-encoder scores (query, passage) pairs
        pairs = [(query, ev.text) for ev in candidates]
        try:
            rerank_scores = self._reranker.predict(pairs, show_progress_bar=False)
            # Update similarity scores with reranker scores
            scored = list(zip(candidates, rerank_scores, strict=False))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = []
            for ev, score in scored[:top_k]:
                # Use reranker score as the new similarity (sigmoid-normalized)
                ev_copy = Evidence(
                    text=ev.text,
                    source=ev.source,
                    title=ev.title,
                    url=ev.url,
                    similarity=float(1 / (1 + np.exp(-score))),  # sigmoid
                    chunk_id=ev.chunk_id,
                )
                reranked.append(ev_copy)
            return reranked
        except Exception as exc:
            logger.warning(f"Reranking failed ({exc}) — returning dense results")
            return candidates[:top_k]

    # ------------------------------------------------------------------
    # Legacy batch_retrieve for backwards compatibility
    # ------------------------------------------------------------------
    def _legacy_batch_retrieve(self, claims: list[str], top_k: int | None = None) -> list[list[Evidence]]:
        """Original FAISS-only batch retrieval (kept for fallback)."""
        if self.index is None:
            return [[] for _ in claims]
        k = top_k or self.config.retrieval.top_k
        embeddings = self.encoder.encode(
            claims,
            normalize_embeddings=self.config.embedding.normalize,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=False,
        )
        query_batch = np.array(embeddings, dtype=np.float32)
        if self._use_numpy_search and self._embedding_matrix is not None:
            scores_batch, indices_batch = self._numpy_search(query_batch, k)
        else:
            with self._faiss_lock:
                scores_batch, indices_batch = self.index.search(query_batch, k)
        return [
            self._build_evidence(scores, indices)
            for scores, indices in zip(scores_batch, indices_batch, strict=False)
        ]

    def _numpy_search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Pure NumPy top-k similarity search as a stability fallback."""
        if self._embedding_matrix is None:
            raise RuntimeError("NumPy search requested but embedding matrix is unavailable")
        sims = query @ self._embedding_matrix.T
        top_idx = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        indices = np.take_along_axis(top_idx, order, axis=1)
        scores = np.take_along_axis(top_scores, order, axis=1)
        return scores.astype(np.float32), indices.astype(np.int64)

    # ------------------------------------------------------------------
    def _build_evidence(self, scores: np.ndarray, indices: np.ndarray) -> list[Evidence]:
        threshold = self.config.retrieval.relevance_threshold
        results: list[Evidence] = []
        for score, idx in zip(scores, indices, strict=False):
            if idx == -1 or score < threshold:
                continue
            meta = self.metadata[int(idx)]
            results.append(
                Evidence(
                    text=meta["text"],
                    source=meta.get("source", "unknown"),
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    similarity=float(score),
                    chunk_id=int(idx),
                )
            )
        return results

    @staticmethod
    def _load_metadata(path: Path) -> list[dict]:
        entries: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        logger.info(f"Loaded {len(entries):,} metadata entries")
        return entries

    # ------------------------------------------------------------------
    # Live Wikipedia API search (6.8M articles, fallback for weak local evidence)
    # ------------------------------------------------------------------
    @staticmethod
    def _wiki_api_search(query: str, max_results: int = 3) -> list[Evidence]:
        """Search Wikipedia API for evidence — full article text, not just intro.

        Gets the FULL article, then extracts the most relevant section by
        finding paragraphs that contain the query's key terms.
        This gives deep evidence from ALL 6.8M English Wikipedia articles.
        """
        # Step 1: Search for relevant article titles
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}&srlimit={max_results}"
            f"&format=json&utf8=1"
        )
        try:
            req = urllib.request.Request(search_url, headers={"User-Agent": "VeriFACT-AI/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []

        # Extract key terms from query for paragraph matching
        stop = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "of", "to", "and", "or", "for", "by", "it", "its", "that", "this", "with", "from", "as", "be", "has", "had", "not"}
        query_terms = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", query)} - stop

        results: list[Evidence] = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            if not snippet or len(snippet) < 20:
                continue

            # Step 2: Get FULL article text (not just intro)
            extract_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}"
                f"&prop=extracts&explaintext=1&format=json&utf8=1"
            )
            try:
                req2 = urllib.request.Request(extract_url, headers={"User-Agent": "VeriFACT-AI/1.0"})
                with urllib.request.urlopen(req2, timeout=6) as resp2:
                    extract_data = json.loads(resp2.read().decode("utf-8"))
                pages = extract_data.get("query", {}).get("pages", {})
                full_text = ""
                for page in pages.values():
                    full_text = page.get("extract", "")
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                full_text = snippet

            if not full_text or len(full_text) < 30:
                full_text = snippet

            # Step 3: Find the MOST RELEVANT paragraph in the full article
            # Split into paragraphs and score each by query term overlap
            paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 40]
            if not paragraphs:
                best_text = full_text[:600]
            else:
                scored = []
                for para in paragraphs:
                    para_words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", para)}
                    overlap = len(query_terms & para_words)
                    scored.append((overlap, para))
                scored.sort(key=lambda x: x[0], reverse=True)

                # Take the top 2 most relevant paragraphs
                best_paras = [p for _, p in scored[:2]]
                best_text = " ".join(best_paras)[:800]

            # Compute actual relevance: what fraction of query terms appear in the evidence?
            evidence_words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", best_text)}
            term_overlap = len(query_terms & evidence_words) / max(len(query_terms), 1)
            # Scale: 0.0 (no overlap) to 0.70 (perfect overlap)
            computed_similarity = min(term_overlap * 0.85, 0.70)

            results.append(Evidence(
                text=best_text,
                source="wikipedia_live",
                title=title,
                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                similarity=round(computed_similarity, 3),
                chunk_id=-1,  # Not from local index
            ))

        return results
