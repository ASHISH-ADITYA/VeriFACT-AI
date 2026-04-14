"""
Annotated Output Generator — Stage 4 of the VeriFactAI pipeline.

Maps verdicts back to the original text and produces:
  - Color-coded HTML with hover tooltips (for Streamlit)
  - Structured JSON report (for API / export)
  - Correction suggestions for contradicted claims

Computes an overall Factuality Score (0–100).
"""

from __future__ import annotations

import html
from typing import Any, Dict, List

from core.claim_decomposer import Claim
from core.llm_client import LLMClient
from utils.helpers import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERDICT_STYLE = {
    "SUPPORTED":    {"bg": "#d4edda", "icon": "&#10004;", "css": "verified"},
    "CONTRADICTED": {"bg": "#f8d7da", "icon": "&#10008;", "css": "contradicted"},
    "UNVERIFIABLE": {"bg": "#fff3cd", "icon": "&#9888;",  "css": "unverifiable"},
    "NO_EVIDENCE":  {"bg": "#e2e3e5", "icon": "&#63;",    "css": "no-evidence"},
}

_CORRECTION_PROMPT = """\
You are a fact-checking assistant. A claim was found to be CONTRADICTED by evidence.
Generate a concise, factual correction (1-2 sentences).

Claim: {claim}
Contradicting evidence: {evidence}

Correction:"""


class AnnotatedOutputGenerator:
    """Produce annotated HTML and structured JSON from verified claims."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm  # optional — used for correction generation

    # ------------------------------------------------------------------
    # HTML annotation
    # ------------------------------------------------------------------
    def generate_html(self, original_text: str, claims: List[Claim]) -> str:
        """Return HTML string with colour-coded claim spans + tooltips."""
        if not claims:
            return html.escape(original_text)

        # Sort claims by char_start to process left-to-right
        annotated_claims = [
            c for c in claims if c.char_start >= 0 and c.char_end > c.char_start
        ]
        annotated_claims.sort(key=lambda c: c.char_start)

        parts: list[str] = []
        cursor = 0

        for claim in annotated_claims:
            # text before this claim span
            if claim.char_start > cursor:
                parts.append(html.escape(original_text[cursor:claim.char_start]))

            style = _VERDICT_STYLE.get(claim.verdict or "NO_EVIDENCE", _VERDICT_STYLE["NO_EVIDENCE"])
            span_text = html.escape(original_text[claim.char_start:claim.char_end])

            ev_snippet = ""
            source_info = ""
            if claim.best_evidence:
                ev_snippet = html.escape(claim.best_evidence.text[:200])
                source_info = html.escape(
                    f"{claim.best_evidence.title} ({claim.best_evidence.source})"
                )

            conf_pct = f"{claim.confidence * 100:.0f}%" if claim.confidence is not None else "N/A"

            tooltip = (
                f"Verdict: {claim.verdict}&#10;"
                f"Confidence: {conf_pct}&#10;"
                f"Evidence: {ev_snippet}&#10;"
                f"Source: {source_info}"
            )

            parts.append(
                f'<span class="{style["css"]}" '
                f'style="background-color:{style["bg"]}; padding:2px 4px; '
                f'border-radius:3px; cursor:help;" '
                f'title="{tooltip}">'
                f'{style["icon"]} {span_text}</span>'
            )
            cursor = claim.char_end

        # remaining text after last claim
        if cursor < len(original_text):
            parts.append(html.escape(original_text[cursor:]))

        return "".join(parts)

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    def generate_json(self, original_text: str, claims: List[Claim]) -> Dict[str, Any]:
        """Structured JSON report suitable for API response or export."""
        supported = sum(1 for c in claims if c.verdict == "SUPPORTED")
        contradicted = sum(1 for c in claims if c.verdict == "CONTRADICTED")
        unverifiable = sum(1 for c in claims if c.verdict == "UNVERIFIABLE")
        no_evidence = sum(1 for c in claims if c.verdict == "NO_EVIDENCE")
        total = len(claims)

        factuality = (supported / total * 100) if total else 0.0

        claims_data: List[Dict[str, Any]] = []
        for c in claims:
            entry: Dict[str, Any] = {
                "id": c.id,
                "claim": c.text,
                "verdict": c.verdict,
                "confidence": c.confidence,
                "claim_type": c.claim_type,
                "source_sentence": c.source_sentence,
                "span": {"start": c.char_start, "end": c.char_end},
            }
            if c.best_evidence:
                entry["evidence"] = {
                    "text": c.best_evidence.text,
                    "source": c.best_evidence.source,
                    "title": c.best_evidence.title,
                    "url": c.best_evidence.url,
                    "similarity": c.best_evidence.similarity,
                }
            if c.correction:
                entry["correction"] = c.correction
            if c.nli_scores:
                entry["nli_scores"] = c.nli_scores
            claims_data.append(entry)

        return {
            "original_text": original_text,
            "factuality_score": round(factuality, 1),
            "total_claims": total,
            "supported": supported,
            "contradicted": contradicted,
            "unverifiable": unverifiable,
            "no_evidence": no_evidence,
            "claims": claims_data,
        }

    # ------------------------------------------------------------------
    # Correction generation (for contradicted claims)
    # ------------------------------------------------------------------
    def generate_corrections(self, claims: List[Claim]) -> None:
        """Fill in .correction for every CONTRADICTED claim (in-place)."""
        if self.llm is None:
            return
        for claim in claims:
            if claim.verdict != "CONTRADICTED" or not claim.best_evidence:
                continue
            try:
                prompt = _CORRECTION_PROMPT.format(
                    claim=claim.text,
                    evidence=claim.best_evidence.text,
                )
                claim.correction = self.llm.generate(
                    user=prompt,
                    system="You are a concise fact-checker.",
                    temperature=0.0,
                    max_tokens=256,
                )
            except Exception as exc:
                logger.warning(f"Correction generation failed for {claim.id}: {exc}")
                claim.correction = None
