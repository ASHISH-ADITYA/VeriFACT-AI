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

from core.semantic_entropy import disagreement_ratio, label_distribution, normalized_entropy
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

        for i in range(max(sc.samples, 1)):
            temperature = min(1.0, sc.temperature_start + i * sc.temperature_step)
            user_prompt = (
                f"Claim: {claim_text}\n\n"
                f"Evidence:\n{evidence_block}\n\n"
                "Return JSON only."
            )
            try:
                raw = self.llm.generate(
                    user=user_prompt,
                    system=_SYSTEM_PROMPT,
                    temperature=temperature,
                    max_tokens=160,
                )
                label = self._parse_label(raw)
                if label is not None:
                    labels.append(label)
            except Exception as exc:
                logger.warning(f"SelfCheck sample failed: {exc}")

        if len(labels) < max(sc.min_valid_samples, 1):
            return None

        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]
        consistency = majority_count / len(labels)

        entropy = normalized_entropy(labels, _CLASSES)
        disagreement = disagreement_ratio(labels)

        uncertainty = (
            sc.uncertainty_entropy_weight * entropy
            + sc.uncertainty_disagreement_weight * disagreement
        )
        uncertainty = min(max(uncertainty, 0.0), 1.0)
        stability = 1.0 - uncertainty

        return {
            "majority_label": majority_label,
            "consistency": round(consistency, 4),
            "entropy": round(entropy, 4),
            "disagreement": round(disagreement, 4),
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
    def _parse_label(raw: str | None) -> str | None:
        if not raw:
            return None

        text = raw.strip()
        try:
            parsed = json.loads(text)
            label = str(parsed.get("label", "")).strip().lower()
            if label in _CLASSES:
                return label
        except Exception:
            pass

        match = re.search(r"supported|contradicted|uncertain", text.lower())
        return match.group(0) if match else None
