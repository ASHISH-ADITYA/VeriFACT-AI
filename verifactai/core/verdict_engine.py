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

        # Step 1b — contradiction-aware evidence mining
        # Track best support AND best contradiction evidence separately.
        # This ensures we never discard strong contradiction signal just
        # because support evidence has higher similarity.
        best_contra_nli = max(nli_results, key=lambda r: r.contradiction)

        # If best contradiction evidence is strong (>0.50) AND its similarity
        # is reasonable (>0.30), this is a real contradiction signal — not noise.
        has_real_contradiction_evidence = (
            best_contra_nli.contradiction > 0.50 and best_contra_nli.evidence.similarity > 0.30
        )

        # Step 2 — aggregate with specificity gate
        #
        # Key insight: NLI models often give high entailment for
        # "topically similar" passages that don't actually verify
        # the specific claim.  We require BOTH high NLI entailment
        # AND high retrieval similarity for a SUPPORTED verdict.
        #
        max_con = max(r.contradiction for r in nli_results)
        best_contra = max(nli_results, key=lambda r: r.contradiction)

        # For entailment, require specificity: entailment * similarity
        # This penalises high-NLI + low-similarity (topical but not specific)
        similarity_gate = 0.45  # tuned: P25 of correct-claim similarity is 0.53

        specific_support_scores = []
        for r in nli_results:
            sim = r.evidence.similarity
            if sim >= similarity_gate and r.entailment > 0.5:
                specific_support_scores.append(r.entailment * sim)
            else:
                specific_support_scores.append(0.0)

        max_specific_ent = max(specific_support_scores) if specific_support_scores else 0.0
        max_raw_ent = max(r.entailment for r in nli_results)

        best_support_idx = (
            max(range(len(specific_support_scores)), key=lambda i: specific_support_scores[i])
            if specific_support_scores
            else 0
        )
        best_support = nli_results[best_support_idx]

        # Step 3 — verdict label (tightened + soft contradiction path)
        nc = self.config.nli

        # CONTRADICTED if:
        #   Hard path: NLI contradiction > threshold AND > raw entailment
        #   Soft path: moderate contradiction (>0.50) AND > specificity-gated entailment
        #   The soft path catches cases with small corpus / indirect evidence
        is_contradicted = (max_con > nc.contradiction_threshold and max_con > max_raw_ent) or (
            max_con > 0.50 and max_con > max_specific_ent
        )

        if is_contradicted:
            label = "CONTRADICTED"
            best_ev = best_contra.evidence
        elif max_specific_ent > nc.entailment_threshold:
            # Only SUPPORTED if evidence is both relevant AND entailing
            label = "SUPPORTED"
            best_ev = best_support.evidence
        elif has_real_contradiction_evidence:
            # Step 3b — contradiction-aware fallback
            # Don't default to UNVERIFIABLE if we have real contradiction evidence.
            # This catches cases where support evidence is weak but contradiction
            # evidence exists — the claim is more likely wrong than unknown.
            label = "CONTRADICTED"
            best_ev = best_contra_nli.evidence
        else:
            label = "UNVERIFIABLE"
            best_ev = best_support.evidence

        # Step 4 — uncertainty signal (semantic-entropy proxy + disagreement)
        uncertainty = self._uncertainty_score(nli_results)
        stability = round(1.0 - uncertainty, 4)

        # Step 5 — calibrated confidence (uses specificity-gated entailment)
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
