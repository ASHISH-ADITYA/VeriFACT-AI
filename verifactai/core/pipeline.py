"""
VeriFactPipeline — End-to-end orchestrator.

Chains: Claim Decomposition → Evidence Retrieval → NLI Verdict →
        Confidence Scoring → Annotated Output.

Provides two entry points:
  verify_text()  — verify pre-existing text (e.g. paste from ChatGPT)
  verify_query() — ask an LLM a question, then verify its answer

Includes in-memory caching to avoid redundant processing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import Config, Profile
from core.llm_client import LLMClient
from core.claim_decomposer import ClaimDecomposer, Claim
from core.evidence_retriever import EvidenceRetriever
from core.verdict_engine import VerdictEngine
from core.annotator import AnnotatedOutputGenerator
from utils.helpers import logger, md5_hash, timed


@dataclass
class VerificationResult:
    """Complete output of a verification run."""
    original_text: str
    llm_query: Optional[str]          # the question that produced this text (if any)
    claims: list                       # List[Claim]
    annotated_html: str
    report_json: Dict[str, Any]
    factuality_score: float            # 0–100
    total_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    no_evidence: int
    processing_time: float             # seconds


class VeriFactPipeline:
    """Production-grade verification pipeline."""

    def __init__(self, config: Config | None = None, profile: Profile | None = None) -> None:
        self.config = config or Config()
        if profile:
            self.config.apply_profile(profile)
        logger.info(
            f"Initialising VeriFactPipeline (profile={self.config.active_profile.value}, "
            f"provider={self.config.llm.provider}, model={self.config.llm.model})"
        )

        self.llm = LLMClient(self.config.llm)
        self.decomposer = ClaimDecomposer(self.llm)
        self.retriever = EvidenceRetriever(self.config)
        self.verdict_engine = VerdictEngine(self.config)
        self.annotator = AnnotatedOutputGenerator(llm=self.llm)

        self._cache: dict[str, VerificationResult] = {}
        logger.info("VeriFactPipeline ready")

    # ------------------------------------------------------------------
    @timed
    def verify_text(self, text: str) -> VerificationResult:
        """Verify pre-existing text (mode: paste LLM output)."""
        cache_key = md5_hash(text)
        if cache_key in self._cache:
            logger.info("Cache hit — returning cached result")
            return self._cache[cache_key]

        start = time.perf_counter()

        # Stage 1 — Claim Decomposition
        claims = self.decomposer.decompose(text)
        logger.info(f"Stage 1 complete: {len(claims)} claims extracted")

        # Stage 2 — Evidence Retrieval (batch)
        claim_texts = [c.text for c in claims]
        evidence_batches = self.retriever.batch_retrieve(claim_texts)
        for claim, evidence in zip(claims, evidence_batches):
            claim.evidence = evidence
        logger.info("Stage 2 complete: evidence retrieved")

        # Stage 3 — NLI Verdict + Confidence
        for claim in claims:
            verdict = self.verdict_engine.judge(claim, claim.evidence)
            claim.verdict = verdict.label
            claim.confidence = verdict.confidence
            claim.best_evidence = verdict.best_evidence
            claim.nli_scores = verdict.nli_scores
            claim.all_nli_results = verdict.all_nli
        logger.info("Stage 3 complete: verdicts assigned")

        # Stage 4 — Corrections for contradicted claims
        self.annotator.generate_corrections(claims)

        # Stage 5 — Annotated output
        annotated_html = self.annotator.generate_html(text, claims)
        report_json = self.annotator.generate_json(text, claims)
        logger.info("Stage 4–5 complete: output annotated")

        elapsed = time.perf_counter() - start

        result = VerificationResult(
            original_text=text,
            llm_query=None,
            claims=claims,
            annotated_html=annotated_html,
            report_json=report_json,
            factuality_score=report_json["factuality_score"],
            total_claims=report_json["total_claims"],
            supported=report_json["supported"],
            contradicted=report_json["contradicted"],
            unverifiable=report_json["unverifiable"],
            no_evidence=report_json["no_evidence"],
            processing_time=round(elapsed, 2),
        )

        self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    @timed
    def verify_query(self, query: str, strict: bool = False) -> VerificationResult:
        """Ask the LLM a question, then verify its answer.

        Args:
            strict: If True (eval mode), use primary provider only and raise on failure.
                    If False (query/demo mode), use fallback chain.
        """
        logger.info(f"Generating LLM response for query: {query[:80]}…")
        llm_response = self.llm.generate(
            user=query,
            system=(
                "You are a knowledgeable assistant. Answer the question factually "
                "and specifically, including concrete details, dates, and names "
                "where relevant."
            ),
            temperature=0.3,
            strict=strict,
        )
        if llm_response is None:
            raise RuntimeError("All LLM providers failed for query generation")
        logger.info(f"LLM response received ({len(llm_response)} chars)")

        result = self.verify_text(llm_response)
        result.llm_query = query
        return result
