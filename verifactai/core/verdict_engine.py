"""
Verdict Engine — Stage 3 of the VeriFactAI pipeline.

For each claim + its retrieved evidence:
  1. Run NLI (DeBERTa-v3) to get entailment / neutral / contradiction scores.
  2. Aggregate across top-k evidence passages.
  3. Compute a calibrated Bayesian confidence score.
  4. Return a structured Verdict.

No circular LLM reasoning — verification uses a dedicated NLI model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.evidence_retriever import Evidence
from utils.helpers import logger, timed

if TYPE_CHECKING:
    from config import Config
    from core.claim_decomposer import Claim


@dataclass
class NLIResult:
    """NLI scores for a single (evidence, claim) pair."""

    evidence: Evidence
    entailment: float
    neutral: float
    contradiction: float


@dataclass
class Verdict:
    """Final judgement on a single claim."""

    label: str  # SUPPORTED | CONTRADICTED | UNVERIFIABLE | NO_EVIDENCE
    confidence: float  # 0.0 – 1.0 calibrated
    uncertainty: float  # 0.0 – 1.0 (higher = less reliable)
    stability: float  # 0.0 – 1.0 (higher = more stable)
    best_evidence: Evidence | None
    nli_scores: dict  # aggregated {entailment, neutral, contradiction}
    all_nli: list[NLIResult]


class VerdictEngine:
    """NLI-based claim verification with Bayesian confidence."""

    # DeBERTa label order: 0=contradiction, 1=neutral, 2=entailment
    _LABEL_IDX = {"contradiction": 0, "neutral": 1, "entailment": 2}

    def __init__(self, config: Config) -> None:
        self.config = config
        nc = config.nli

        logger.info(f"Loading NLI model: {nc.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(nc.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nc.model_name)
        self.model.eval()

        # Use MPS on Apple Silicon if available, else CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("NLI model using Apple MPS acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("NLI model using CPU")
        self.model.to(self.device)

        self._source_reliability = config.confidence.source_reliability
        logger.info("VerdictEngine ready")

    # ------------------------------------------------------------------
    @timed
    def judge(self, claim: Claim, evidence_list: list[Evidence]) -> Verdict:
        """Produce a verdict for one claim against its retrieved evidence."""

        # ── Rule-based pre-check: catch obvious impossibilities ──
        from core.fact_rules import check_rules

        rule_violation = check_rules(claim.text)
        if rule_violation is not None:
            logger.info(
                f"Rule override [{rule_violation.rule_name}]: {claim.text[:60]} → CONTRADICTED"
            )
            # Create a synthetic evidence from the rule
            rule_evidence = Evidence(
                text=rule_violation.correct_fact,
                source="factual_rule",
                title=f"Rule: {rule_violation.rule_name}",
                url="",
                similarity=1.0,
                chunk_id=-1,
            )
            return Verdict(
                label="CONTRADICTED",
                confidence=0.95,
                uncertainty=0.05,
                stability=0.95,
                best_evidence=rule_evidence,
                nli_scores={
                    "entailment": 0.0,
                    "contradiction": 1.0,
                    "neutral": 0.0,
                    "rule_override": True,
                    "rule_name": rule_violation.rule_name,
                    "rule_reason": rule_violation.reason,
                },
                all_nli=[],
            )

        if not evidence_list:
            return Verdict(
                label="NO_EVIDENCE",
                confidence=0.0,
                uncertainty=1.0,
                stability=0.0,
                best_evidence=None,
                nli_scores={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
                all_nli=[],
            )

        # Step 1 — run NLI for each (evidence, claim) pair
        nli_results = self._batch_nli(claim.text, evidence_list)

        # Steps 2–5 — apply verdict logic (shared with batch_judge)
        return self._apply_verdict_logic(claim, evidence_list, nli_results)

    def _uncertainty_score(self, nli_results: list[NLIResult]) -> float:
        """
        Estimate uncertainty from NLI distribution entropy + inter-evidence disagreement.

        This is a practical semantic-entropy proxy: if verdict probability mass is diffuse
        and evidence pairs disagree, uncertainty should increase.
        """
        if not nli_results:
            return 1.0

        prob_matrix = np.array(
            [[r.entailment, r.neutral, r.contradiction] for r in nli_results],
            dtype=np.float32,
        )

        # Mean class probabilities across retrieved evidence.
        mean_probs = np.mean(prob_matrix, axis=0)
        mean_probs = np.clip(mean_probs, 1e-8, 1.0)
        mean_probs = mean_probs / np.sum(mean_probs)

        # Normalized entropy in [0,1].
        entropy = float(-np.sum(mean_probs * np.log(mean_probs)) / np.log(3.0))

        # Evidence disagreement based on contradiction/entailment spread.
        disagreement = float(0.5 * np.std(prob_matrix[:, 0]) + 0.5 * np.std(prob_matrix[:, 2]))
        disagreement = min(max(disagreement * 2.0, 0.0), 1.0)

        cw = self.config.confidence
        uncertainty = (
            cw.uncertainty_entropy_weight * entropy
            + cw.uncertainty_disagreement_weight * disagreement
        )
        return round(min(max(uncertainty, 0.0), 1.0), 4)

    # ------------------------------------------------------------------
    # Cross-claim batch judging
    # ------------------------------------------------------------------
    def batch_judge(
        self, claims_with_evidence: list[tuple[Claim, list[Evidence]]]
    ) -> list[Verdict]:
        """Produce verdicts for multiple claims in one batched NLI forward pass.

        Instead of calling judge() per claim (one forward pass each), this
        collects ALL (premise, hypothesis) pairs across all claims, runs a
        single tokenize + forward pass, then splits results back per-claim
        and applies the same verdict logic.
        """
        from core.fact_rules import check_rules

        # Pre-check: separate rule-overridden and no-evidence claims
        verdicts: list[Verdict | None] = [None] * len(claims_with_evidence)
        nli_work: list[tuple[int, Claim, list[Evidence]]] = []  # (original_idx, claim, evidence)

        for idx, (claim, evidence_list) in enumerate(claims_with_evidence):
            rule_violation = check_rules(claim.text)
            if rule_violation is not None:
                logger.info(
                    f"Rule override [{rule_violation.rule_name}]: {claim.text[:60]} → CONTRADICTED"
                )
                rule_evidence = Evidence(
                    text=rule_violation.correct_fact,
                    source="factual_rule",
                    title=f"Rule: {rule_violation.rule_name}",
                    url="",
                    similarity=1.0,
                    chunk_id=-1,
                )
                verdicts[idx] = Verdict(
                    label="CONTRADICTED",
                    confidence=0.95,
                    uncertainty=0.05,
                    stability=0.95,
                    best_evidence=rule_evidence,
                    nli_scores={
                        "entailment": 0.0,
                        "contradiction": 1.0,
                        "neutral": 0.0,
                        "rule_override": True,
                        "rule_name": rule_violation.rule_name,
                        "rule_reason": rule_violation.reason,
                    },
                    all_nli=[],
                )
            elif not evidence_list:
                verdicts[idx] = Verdict(
                    label="NO_EVIDENCE",
                    confidence=0.0,
                    uncertainty=1.0,
                    stability=0.0,
                    best_evidence=None,
                    nli_scores={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
                    all_nli=[],
                )
            else:
                nli_work.append((idx, claim, evidence_list))

        if not nli_work:
            return [v for v in verdicts if v is not None]  # type: ignore[misc]

        # Collect ALL (premise, hypothesis) pairs across all claims
        all_premises: list[str] = []
        all_hypotheses: list[str] = []
        all_evidence_refs: list[Evidence] = []
        # Track boundaries: (start_offset, count) for each work item
        boundaries: list[tuple[int, int]] = []

        for _idx, claim, evidence_list in nli_work:
            start = len(all_premises)
            for ev in evidence_list:
                all_premises.append(ev.text)
                all_hypotheses.append(claim.text)
                all_evidence_refs.append(ev)
            boundaries.append((start, len(evidence_list)))

        # Single tokenize + forward pass
        inputs = self.tokenizer(
            all_premises,
            all_hypotheses,
            padding=True,
            truncation=True,
            max_length=self.config.nli.max_input_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            all_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Split results back per-claim and apply verdict logic
        for work_i, (orig_idx, claim, evidence_list) in enumerate(nli_work):
            start, count = boundaries[work_i]
            nli_results: list[NLIResult] = []
            for j in range(count):
                p = all_probs[start + j]
                nli_results.append(
                    NLIResult(
                        evidence=all_evidence_refs[start + j],
                        contradiction=float(p[0]),
                        neutral=float(p[1]),
                        entailment=float(p[2]),
                    )
                )
            verdicts[orig_idx] = self._apply_verdict_logic(claim, evidence_list, nli_results)

        return [v for v in verdicts if v is not None]  # type: ignore[misc]

    def _apply_verdict_logic(
        self, claim: Claim, evidence_list: list[Evidence], nli_results: list[NLIResult]
    ) -> Verdict:
        """Apply verdict logic with strong prior toward SUPPORTED.

        Key principle: most chatbot responses are factually correct.
        Only flag CONTRADICTED when evidence is STRONG and RELEVANT.
        Default to SUPPORTED (not UNVERIFIABLE) when evidence is weak.

        Extracted from judge() so both judge() and batch_judge() share the same logic.
        """
        # ── Gather NLI signals ──
        max_con = max(r.contradiction for r in nli_results)
        max_raw_ent = max(r.entailment for r in nli_results)
        best_contra = max(nli_results, key=lambda r: r.contradiction)
        max_similarity = max(r.evidence.similarity for r in nli_results)

        # Specificity-gated support: high entailment + relevant evidence
        similarity_gate = 0.40
        specific_support_scores = []
        for r in nli_results:
            sim = r.evidence.similarity
            if sim >= similarity_gate and r.entailment > 0.4:
                specific_support_scores.append(r.entailment * sim)
            else:
                specific_support_scores.append(0.0)

        max_specific_ent = max(specific_support_scores) if specific_support_scores else 0.0

        best_support_idx = (
            max(range(len(specific_support_scores)), key=lambda i: specific_support_scores[i])
            if specific_support_scores
            else 0
        )
        best_support = nli_results[best_support_idx]

        nc = self.config.nli

        # ── CONTRADICTED: only with STRONG + RELEVANT evidence ──
        # Hard path: very high NLI contradiction + beats entailment + relevant evidence
        hard_contra = (
            max_con > nc.contradiction_threshold
            and max_con > max_raw_ent
            and best_contra.evidence.similarity > 0.45
        )

        # Soft path: moderate contradiction but evidence must be highly relevant
        soft_contra = (
            max_con > 0.65
            and max_con > max_specific_ent + 0.15  # must beat support by clear margin
            and best_contra.evidence.similarity > 0.50
        )

        is_contradicted = hard_contra or soft_contra

        # ── SUPPORTED: generous — most facts are correct ──
        # Strong support: specificity-gated entailment passes threshold
        strong_support = max_specific_ent > nc.entailment_threshold

        # Moderate support: raw entailment is decent with some relevant evidence
        moderate_support = max_raw_ent > 0.40 and max_similarity > 0.35

        # Weak support: evidence exists and doesn't contradict
        # This is the key change — when evidence is ambiguous, lean SUPPORTED
        weak_support = max_similarity > 0.30 and max_con < 0.50

        # ── Decision order: CONTRADICTED → SUPPORTED → UNVERIFIABLE ──
        if is_contradicted:
            label = "CONTRADICTED"
            best_ev = best_contra.evidence
        elif strong_support or moderate_support:
            label = "SUPPORTED"
            best_ev = best_support.evidence
        elif weak_support:
            # Default: evidence exists, no contradiction → trust the chatbot
            label = "SUPPORTED"
            best_ev = best_support.evidence
        else:
            # Only truly unverifiable: no relevant evidence at all
            label = "UNVERIFIABLE"
            best_ev = best_support.evidence

        # Uncertainty
        uncertainty = self._uncertainty_score(nli_results)
        stability = round(1.0 - uncertainty, 4)

        # Confidence
        confidence = self._bayesian_confidence(
            nli_results, evidence_list, max_specific_ent, max_con, uncertainty
        )

        return Verdict(
            label=label,
            confidence=confidence,
            uncertainty=uncertainty,
            stability=stability,
            best_evidence=best_ev,
            nli_scores={
                "entailment": round(max_specific_ent, 4),
                "raw_entailment": round(max_raw_ent, 4),
                "neutral": round(max(0, 1 - max_raw_ent - max_con), 4),
                "contradiction": round(max_con, 4),
                "uncertainty": uncertainty,
                "stability": stability,
            },
            all_nli=nli_results,
        )

    # ------------------------------------------------------------------
    # NLI inference
    # ------------------------------------------------------------------
    def _batch_nli(self, claim_text: str, evidence_list: list[Evidence]) -> list[NLIResult]:
        """Run NLI on all (evidence, claim) pairs in one batched forward pass."""
        premises = [ev.text for ev in evidence_list]
        hypotheses = [claim_text] * len(evidence_list)

        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.config.nli.max_input_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: list[NLIResult] = []
        for ev, p in zip(evidence_list, probs, strict=False):
            results.append(
                NLIResult(
                    evidence=ev,
                    contradiction=float(p[0]),
                    neutral=float(p[1]),
                    entailment=float(p[2]),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Bayesian confidence scoring
    # ------------------------------------------------------------------
    def _bayesian_confidence(
        self,
        nli_results: list[NLIResult],
        evidence_list: list[Evidence],
        max_ent: float,
        max_con: float,
        uncertainty: float,
    ) -> float:
        """
        Fuse four signals:
          1. NLI support score (max_ent − max_con normalised to 0-1)
          2. Retrieval relevance (best cosine similarity)
          3. Source reliability prior
          4. Cross-reference agreement (fraction of evidence that entails)
          5. Uncertainty stability bonus (1 - uncertainty)
        """
        w = self.config.confidence

        # Signal 1: NLI support
        nli_support = (max_ent - max_con + 1.0) / 2.0  # map [-1,1] → [0,1]

        # Signal 2: retrieval relevance
        retrieval_rel = max((ev.similarity for ev in evidence_list), default=0.0)

        # Signal 3: source reliability
        src_scores = [self._source_reliability.get(ev.source, 0.5) for ev in evidence_list[:3]]
        src_rel = float(np.mean(src_scores)) if src_scores else 0.5

        # Signal 4: cross-reference agreement
        supporting = sum(
            1 for r in nli_results if r.entailment > r.contradiction and r.entailment > 0.5
        )
        cross_ref = supporting / len(nli_results) if nli_results else 0.0

        # Signal 5: uncertainty-aware stability
        stability = 1.0 - uncertainty

        score = (
            w.w_nli * nli_support
            + w.w_retrieval * retrieval_rel
            + w.w_source * src_rel
            + w.w_cross_ref * cross_ref
            + w.w_uncertainty * stability
        )
        return round(min(max(score, 0.0), 1.0), 4)
