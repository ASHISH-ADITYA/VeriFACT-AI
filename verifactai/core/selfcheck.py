"""
SelfCheck-style consistency sampling for claim verification.

This module queries the configured local LLM multiple times with slight
sampling variation, then estimates uncertainty via semantic entropy and
cross-sample disagreement.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import TYPE_CHECKING

from core.semantic_entropy import (
    cluster_entropy,
    disagreement_ratio,
    label_distribution,
    normalized_entropy,
)
from utils.helpers import logger

if TYPE_CHECKING:
    from config import Config
    from core.evidence_retriever import Evidence
    from core.llm_client import LLMClient

_CLASSES = ["supported", "contradicted", "uncertain"]

_SYSTEM_PROMPT = """You are a strict factual verifier.
Read the claim and evidence snippets and return JSON only:
{
  "label": "supported|contradicted|uncertain",
  "rationale": "one short sentence"
}
Use 'supported' only if the evidence directly supports the claim.
Use 'contradicted' if evidence clearly conflicts.
Use 'uncertain' if evidence is weak, missing, or mixed."""


class SelfCheckScorer:
    """Estimate claim-level consistency uncertainty using repeated judgments."""

    def __init__(self, llm: LLMClient | None, config: Config) -> None:
        self.llm = llm
        self.config = config

    def score_claim(self, claim_text: str, evidence: list[Evidence]) -> dict | None:
        """Return self-check metrics, or None when unavailable."""
        sc = self.config.selfcheck
        if not sc.enabled or self.llm is None or not evidence:
            return None

        evidence_block = self._build_evidence_block(evidence[: max(sc.max_evidence, 1)])
        labels: list[str] = []
        rationales: list[str] = []

        for i in range(max(sc.samples, 1)):
            temperature = min(1.0, sc.temperature_start + i * sc.temperature_step)
            user_prompt = f"Claim: {claim_text}\n\nEvidence:\n{evidence_block}\n\nReturn JSON only."
            try:
                raw = self.llm.generate(
                    user=user_prompt,
                    system=_SYSTEM_PROMPT,
                    temperature=temperature,
                    max_tokens=160,
                )
                label, rationale = self._parse_sample(raw)
                if label is not None:
                    labels.append(label)
                if rationale:
                    rationales.append(rationale)
            except Exception as exc:
                logger.warning(f"SelfCheck sample failed: {exc}")

        if len(labels) < max(sc.min_valid_samples, 1):
            return None

        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]
        consistency = majority_count / len(labels)

        entropy = normalized_entropy(labels, _CLASSES)
        disagreement = disagreement_ratio(labels)
        semantic_entropy = cluster_entropy(rationales, threshold=0.5)

        ew = max(sc.uncertainty_entropy_weight, 0.0)
        dw = max(sc.uncertainty_disagreement_weight, 0.0)
        sw = max(sc.semantic_cluster_weight, 0.0)
        denom = max(ew + dw + sw, 1e-8)

        uncertainty = (ew * entropy + dw * disagreement + sw * semantic_entropy) / denom
        uncertainty = min(max(uncertainty, 0.0), 1.0)
        stability = 1.0 - uncertainty

        return {
            "majority_label": majority_label,
            "consistency": round(consistency, 4),
            "entropy": round(entropy, 4),
            "disagreement": round(disagreement, 4),
            "semantic_cluster_entropy": round(semantic_entropy, 4),
            "uncertainty": round(uncertainty, 4),
            "stability": round(stability, 4),
            "distribution": label_distribution(labels, _CLASSES),
            "valid_samples": len(labels),
        }

    @staticmethod
    def _build_evidence_block(evidence: list[Evidence]) -> str:
        lines: list[str] = []
        for i, ev in enumerate(evidence, start=1):
            lines.append(
                f"[{i}] source={ev.source}; title={ev.title}; similarity={ev.similarity:.3f}\n"
                f"{ev.text[:600]}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _parse_sample(raw: str | None) -> tuple[str | None, str | None]:
        if not raw:
            return (None, None)

        text = raw.strip()
        try:
            parsed = json.loads(text)
            label = str(parsed.get("label", "")).strip().lower()
            rationale = str(parsed.get("rationale", "")).strip()
            if label in _CLASSES:
                return (label, rationale or None)
        except Exception:
            pass

        match = re.search(r"supported|contradicted|uncertain", text.lower())
        label = match.group(0) if match else None
        return (label, None)
