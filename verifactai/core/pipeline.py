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
from typing import Any

from config import Config, Profile
from core.annotator import AnnotatedOutputGenerator
from core.claim_decomposer import ClaimDecomposer
from core.evidence_retriever import EvidenceRetriever
from core.llm_client import LLMClient
from core.selfcheck import SelfCheckScorer
from core.verdict_engine import VerdictEngine
from utils.helpers import logger, md5_hash, timed


@dataclass
class VerificationResult:
    """Complete output of a verification run."""

    original_text: str
    llm_query: str | None  # the question that produced this text (if any)
    claims: list  # List[Claim]
    annotated_html: str
    report_json: dict[str, Any]
    factuality_score: float  # 0–100
    total_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    no_evidence: int
    processing_time: float  # seconds


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
        self.verdict_engine = VerdictEngine(self.config, llm_client=self.llm)
        self.selfcheck = SelfCheckScorer(self.llm, self.config, verdict_engine=self.verdict_engine)
        self.annotator = AnnotatedOutputGenerator(
            llm=self.llm,
            reflexion_enabled=self.config.reflexion.enabled,
            reflexion_rounds=self.config.reflexion.max_rounds,
            constitutional_enabled=self.config.constitutional.enabled,
            constitutional_rounds=self.config.constitutional.max_rounds,
        )

        self._cache: dict[str, VerificationResult] = {}
        logger.info("VeriFactPipeline ready")

    # ------------------------------------------------------------------
    def _verify_text_impl(self, text: str, fast_mode: bool = False) -> VerificationResult:
        """Verify pre-existing text (mode: paste LLM output).

        fast_mode skips expensive refinement loops for extension-first latency.
        """
        cache_key = f"{'fast' if fast_mode else 'full'}:{md5_hash(text)}"
        if cache_key in self._cache:
            logger.info("Cache hit — returning cached result")
            return self._cache[cache_key]

        start = time.perf_counter()

        from core.evidence_retriever import Evidence
        from core.fact_rules import check_rules
        from core.claim_decomposer import Claim

        # Stage 0 — Whole-text rule pre-check: if the raw input matches a rule,
        # synthesise a single claim immediately (avoids decomposer dropping short
        # sentences like "The Earth is flat.").
        whole_text_rule = check_rules(text)
        if whole_text_rule is not None:
            synth = Claim(
                id="c-rule-0",
                text=text.strip(),
                source_sentence=text.strip(),
                claim_type="entity_fact",
                char_start=0,
                char_end=len(text),
            )
            synth.evidence = [Evidence(
                text=whole_text_rule.correct_fact, source="factual_rule",
                title=f"Rule: {whole_text_rule.rule_name}", url="", similarity=1.0, chunk_id=-1,
            )]
            synth.verdict = "CONTRADICTED"
            synth.confidence = 0.95
            synth.uncertainty = 0.05
            synth.stability = 0.95
            synth.best_evidence = synth.evidence[0]
            synth.nli_scores = {"rule_override": True, "rule_name": whole_text_rule.rule_name,
                                "rule_reason": whole_text_rule.reason}
            synth.all_nli_results = []
            claims = [synth]
            rule_resolved = {synth.id}
            logger.info(f"Stage 0 whole-text rule [{whole_text_rule.rule_name}]: forced CONTRADICTED")
        else:
            # Stage 1 — Claim Decomposition (normal path)
            claims = self.decomposer.decompose(text)
            logger.info(f"Stage 1 complete: {len(claims)} claims extracted")

            # Stage 2a — Per-claim rule check
            rule_resolved = set()
            for claim in claims:
                rule = check_rules(claim.text)
                if rule:
                    logger.info(f"Rule fast-pass [{rule.rule_name}]: {claim.text[:50]} → CONTRADICTED")
                    claim.evidence = [Evidence(
                        text=rule.correct_fact, source="factual_rule",
                        title=f"Rule: {rule.rule_name}", url="", similarity=1.0, chunk_id=-1,
                    )]
                    claim.verdict = "CONTRADICTED"
                    claim.confidence = 0.95
                    claim.uncertainty = 0.05
                    claim.stability = 0.95
                    claim.best_evidence = claim.evidence[0]
                    claim.nli_scores = {"rule_override": True, "rule_name": rule.rule_name, "rule_reason": rule.reason}
                    claim.all_nli_results = []
                    rule_resolved.add(claim.id)

        # Stage 2b — Evidence Retrieval (only for unresolved claims)
        unresolved = [c for c in claims if c.id not in rule_resolved]
        if unresolved:
            unresolved_texts = [c.text for c in unresolved]
            evidence_batches = self.retriever.batch_retrieve(unresolved_texts)
            for claim, evidence in zip(unresolved, evidence_batches, strict=False):
                claim.evidence = evidence
        logger.info(f"Stage 2 complete: {len(rule_resolved)} ruled, {len(unresolved)} need NLI")

        # Stage 3 — NLI Verdict (only for unresolved claims, batched)
        if unresolved:
            claims_with_evidence = [(claim, claim.evidence) for claim in unresolved]
            verdicts = self.verdict_engine.batch_judge(claims_with_evidence)
        else:
            verdicts = []
        for claim, verdict in zip(unresolved, verdicts, strict=False):
            claim.verdict = verdict.label
            claim.confidence = verdict.confidence
            claim.uncertainty = verdict.uncertainty
            claim.stability = verdict.stability
            claim.best_evidence = verdict.best_evidence
            claim.nli_scores = verdict.nli_scores
            claim.all_nli_results = verdict.all_nli

            selfcheck = None if fast_mode else self.selfcheck.score_claim(claim.text, claim.evidence)
            if selfcheck:
                blend = min(max(self.config.selfcheck.confidence_blend_weight, 0.0), 1.0)
                claim.confidence = round(
                    (1.0 - blend) * float(claim.confidence)
                    + blend * float(selfcheck["consistency"]),
                    4,
                )
                claim.uncertainty = round(
                    (float(claim.uncertainty) + float(selfcheck["uncertainty"])) / 2.0,
                    4,
                )
                claim.stability = round(1.0 - float(claim.uncertainty), 4)
                if claim.nli_scores is None:
                    claim.nli_scores = {}
                claim.nli_scores.update(
                    {
                        "selfcheck_consistency": selfcheck["consistency"],
                        "selfcheck_entropy": selfcheck["entropy"],
                        "selfcheck_disagreement": selfcheck["disagreement"],
                        "selfcheck_semantic_cluster_entropy": selfcheck["semantic_cluster_entropy"],
                        "selfcheck_distribution": selfcheck["distribution"],
                        "selfcheck_majority_label": selfcheck["majority_label"],
                        "selfcheck_valid_samples": selfcheck["valid_samples"],
                    }
                )
        logger.info("Stage 3 complete: verdicts assigned")

        # Stage 4 — Corrections for contradicted claims
        if not fast_mode:
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
    def verify_text_streaming(self, text: str, fast_mode: bool = False):
        """Generator that yields intermediate progress dicts during verification.

        Yields dicts with an ``event`` key (status | claim | verdict | complete).
        The final yield is always ``complete`` with the full VerificationResult.
        """
        cache_key = f"{'fast' if fast_mode else 'full'}:{md5_hash(text)}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            yield {"event": "complete", "data": cached}
            return

        start = time.perf_counter()

        # Stage 1 — Claim Decomposition
        yield {"event": "status", "data": {"stage": "decomposing", "message": "Extracting claims..."}}
        claims = self.decomposer.decompose(text)
        logger.info(f"Stage 1 complete: {len(claims)} claims extracted")

        total = len(claims)
        for idx, claim in enumerate(claims):
            yield {"event": "claim", "data": {"claim": claim.text, "index": idx, "total": total}}

        # Stage 2a — Fast rule-check pass
        from core.evidence_retriever import Evidence
        from core.fact_rules import check_rules

        rule_resolved = set()
        for claim in claims:
            rule = check_rules(claim.text)
            if rule:
                claim.evidence = [Evidence(
                    text=rule.correct_fact, source="factual_rule",
                    title=f"Rule: {rule.rule_name}", url="", similarity=1.0, chunk_id=-1,
                )]
                claim.verdict = "CONTRADICTED"
                claim.confidence = 0.95
                claim.uncertainty = 0.05
                claim.stability = 0.95
                claim.best_evidence = claim.evidence[0]
                claim.nli_scores = {"rule_override": True, "rule_name": rule.rule_name}
                claim.all_nli_results = []
                rule_resolved.add(claim.id)
                yield {
                    "event": "verdict",
                    "data": {"claim": claim.text, "verdict": "CONTRADICTED", "confidence": 0.95,
                             "evidence": rule.correct_fact},
                }

        # Stage 2b — Evidence Retrieval (only unresolved)
        unresolved = [c for c in claims if c.id not in rule_resolved]
        if unresolved:
            yield {"event": "status", "data": {"stage": "retrieving", "message": f"Retrieving evidence for {len(unresolved)} claims..."}}
            unresolved_texts = [c.text for c in unresolved]
            evidence_batches = self.retriever.batch_retrieve(unresolved_texts)
            for claim, evidence in zip(unresolved, evidence_batches, strict=False):
                claim.evidence = evidence
        logger.info(f"Stage 2 complete: {len(rule_resolved)} ruled, {len(unresolved)} need NLI")

        # Stage 3 — NLI Verdict (only unresolved, batched)
        if unresolved:
            yield {"event": "status", "data": {"stage": "judging", "message": "Assigning verdicts..."}}
            claims_with_evidence = [(claim, claim.evidence) for claim in unresolved]
            verdicts = self.verdict_engine.batch_judge(claims_with_evidence)
        else:
            verdicts = []
        for claim, verdict in zip(unresolved, verdicts, strict=False):
            claim.verdict = verdict.label
            claim.confidence = verdict.confidence
            claim.uncertainty = verdict.uncertainty
            claim.stability = verdict.stability
            claim.best_evidence = verdict.best_evidence
            claim.nli_scores = verdict.nli_scores
            claim.all_nli_results = verdict.all_nli

            selfcheck = None if fast_mode else self.selfcheck.score_claim(claim.text, claim.evidence)
            if selfcheck:
                blend = min(max(self.config.selfcheck.confidence_blend_weight, 0.0), 1.0)
                claim.confidence = round(
                    (1.0 - blend) * float(claim.confidence)
                    + blend * float(selfcheck["consistency"]),
                    4,
                )
                claim.uncertainty = round(
                    (float(claim.uncertainty) + float(selfcheck["uncertainty"])) / 2.0,
                    4,
                )
                claim.stability = round(1.0 - float(claim.uncertainty), 4)
                if claim.nli_scores is None:
                    claim.nli_scores = {}
                claim.nli_scores.update(
                    {
                        "selfcheck_consistency": selfcheck["consistency"],
                        "selfcheck_entropy": selfcheck["entropy"],
                        "selfcheck_disagreement": selfcheck["disagreement"],
                        "selfcheck_semantic_cluster_entropy": selfcheck["semantic_cluster_entropy"],
                        "selfcheck_distribution": selfcheck["distribution"],
                        "selfcheck_majority_label": selfcheck["majority_label"],
                        "selfcheck_valid_samples": selfcheck["valid_samples"],
                    }
                )

            yield {
                "event": "verdict",
                "data": {
                    "claim": claim.text,
                    "verdict": claim.verdict,
                    "confidence": float(claim.confidence or 0.0),
                    "evidence": (claim.best_evidence.text[:220] if claim.best_evidence else ""),
                },
            }
        logger.info("Stage 3 complete: verdicts assigned")

        # Stage 4 — Corrections for contradicted claims
        if not fast_mode:
            self.annotator.generate_corrections(claims)

        # Stage 5 — Annotated output
        yield {"event": "status", "data": {"stage": "annotating", "message": "Generating report..."}}
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
        yield {"event": "complete", "data": result}

    # ------------------------------------------------------------------
    @timed
    def verify_text_fast(self, text: str) -> VerificationResult:
        """Lightweight fast-path for the browser extension.

        Runs ONLY: spaCy decomposition + rule check + FAISS retrieval
        (no BM25, no reranker) + batch NLI verdict (no selfcheck).
        Skips: corrections, reflexion, constitutional, HTML generation.

        Target: <5s for 3 claims on HF free CPU.
        """
        cache_key = f"ultrafast:{md5_hash(text)}"
        if cache_key in self._cache:
            logger.info("Cache hit — returning cached fast result")
            return self._cache[cache_key]

        start = time.perf_counter()

        # Stage 1 — Claim Decomposition (spaCy fallback, fast, capped at 8)
        claims = self.decomposer.decompose(text)[:8]
        logger.info(f"[fast] Stage 1: {len(claims)} claims extracted (capped at 8)")

        # Stage 2a — Rule check (instant)
        from core.evidence_retriever import Evidence
        from core.fact_rules import check_rules

        rule_resolved: set[Any] = set()
        for claim in claims:
            rule = check_rules(claim.text)
            if rule:
                claim.evidence = [Evidence(
                    text=rule.correct_fact, source="factual_rule",
                    title=f"Rule: {rule.rule_name}", url="", similarity=1.0, chunk_id=-1,
                )]
                claim.verdict = "CONTRADICTED"
                claim.confidence = 0.95
                claim.uncertainty = 0.05
                claim.stability = 0.95
                claim.best_evidence = claim.evidence[0]
                claim.nli_scores = {
                    "rule_override": True,
                    "rule_name": rule.rule_name,
                    "rule_reason": rule.reason,
                }
                claim.all_nli_results = []
                rule_resolved.add(claim.id)

        # Stage 2b — FAISS-only retrieval (no BM25, no reranker)
        unresolved = [c for c in claims if c.id not in rule_resolved]
        if unresolved:
            unresolved_texts = [c.text for c in unresolved]
            evidence_batches = self.retriever.fast_retrieve(unresolved_texts)
            for claim, evidence in zip(unresolved, evidence_batches, strict=False):
                claim.evidence = evidence
        logger.info(
            f"[fast] Stage 2: {len(rule_resolved)} ruled, {len(unresolved)} FAISS-only"
        )

        # Stage 3 — Batch NLI verdict (no selfcheck)
        if unresolved:
            claims_with_evidence = [(claim, claim.evidence) for claim in unresolved]
            verdicts = self.verdict_engine.batch_judge(claims_with_evidence)
        else:
            verdicts = []
        for claim, verdict in zip(unresolved, verdicts, strict=False):
            claim.verdict = verdict.label
            claim.confidence = verdict.confidence
            claim.uncertainty = verdict.uncertainty
            claim.stability = verdict.stability
            claim.best_evidence = verdict.best_evidence
            claim.nli_scores = verdict.nli_scores
            claim.all_nli_results = verdict.all_nli
        logger.info("[fast] Stage 3: verdicts assigned (no selfcheck)")

        # Stage 4 — JSON report only (skip corrections, skip HTML)
        report_json = self.annotator.generate_json(text, claims)

        elapsed = time.perf_counter() - start

        result = VerificationResult(
            original_text=text,
            llm_query=None,
            claims=claims,
            annotated_html="",  # skipped for speed
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
    def verify_text(self, text: str, fast_mode: bool = False) -> VerificationResult:
        """Public verification API.

        Args:
            fast_mode: Skip expensive refinement stages for lower-latency requests.
        """
        return self._verify_text_impl(text, fast_mode=fast_mode)

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
