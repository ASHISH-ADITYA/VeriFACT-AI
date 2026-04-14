"""
Evidence Retriever — Stage 2 of the VeriFactAI pipeline.

Dense retrieval over a pre-built FAISS index of trusted knowledge sources
(Wikipedia, PubMed). Returns top-k evidence passages per claim with metadata.

Supports both single-claim and batch retrieval for throughput.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
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


class EvidenceRetriever:
    """FAISS-backed dense retriever with sentence-transformer encoding."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._faiss_lock = threading.Lock()
        self._use_numpy_search = os.environ.get("VERIFACTAI_USE_NUMPY_SEARCH", "0") == "1"
        self._embedding_matrix: np.ndarray | None = None
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
            # For IndexFlat* we can reconstruct vectors and avoid FAISS search calls.
            self._embedding_matrix = self.index.reconstruct_n(0, self.index.ntotal).astype(
                np.float32
            )
            if self.config.embedding.normalize:
                norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
                self._embedding_matrix = self._embedding_matrix / np.clip(norms, 1e-12, None)

        logger.info(f"Loading metadata from {meta_path}")
        self.metadata = self._load_metadata(meta_path)

    # ------------------------------------------------------------------
    @timed
    def retrieve(self, claim_text: str, top_k: int | None = None) -> list[Evidence]:
        """Retrieve evidence for a single claim."""
        if self.index is None:
            return []
        k = top_k or self.config.retrieval.top_k
        embedding = self.encoder.encode(
            [claim_text],
            normalize_embeddings=self.config.embedding.normalize,
        )
        query = np.array(embedding, dtype=np.float32)
        if self._use_numpy_search and self._embedding_matrix is not None:
            scores, indices = self._numpy_search(query, k)
        else:
            with self._faiss_lock:
                scores, indices = self.index.search(query, k)
        return self._build_evidence(scores[0], indices[0])

    @timed
    def batch_retrieve(self, claims: list[str], top_k: int | None = None) -> list[list[Evidence]]:
        """Batch retrieval for multiple claims — single FAISS call."""
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
