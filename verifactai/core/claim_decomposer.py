"""
Claim Decomposer — Stage 1 of the VeriFactAI pipeline.

Breaks LLM-generated text into atomic, independently verifiable factual claims.
Primary: LLM-based extraction with structured JSON output.
Fallback: spaCy sentence segmentation when LLM is unavailable.

Grounded in: FActScore (Min et al., 2023) methodology.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import spacy

from core.llm_client import LLMClient
from utils.helpers import logger, timed

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A single atomic factual claim extracted from LLM output."""
    id: str                              # stable identifier e.g. "c-0", "c-1"
    text: str                            # self-contained factual statement
    source_sentence: str                 # original sentence it came from
    claim_type: str = "entity_fact"      # entity_fact|numerical|temporal|causal|relational
    char_start: int = -1                 # start offset in original text
    char_end: int = -1                   # end offset in original text
    # Filled downstream
    evidence: list = field(default_factory=list)
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    best_evidence: Optional[object] = None
    nli_scores: Optional[dict] = None
    all_nli_results: Optional[list] = None
    uncertainty: Optional[float] = None
    stability: Optional[float] = None
    correction: Optional[str] = None


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise factual claim extractor for an automated fact-checking system.
Your job: decompose the user-provided text into atomic, independently verifiable
factual claims.

RULES
1. Each claim MUST be a single factual assertion (one fact per claim).
2. Each claim MUST be self-contained — understandable without the original text.
3. RESOLVE all pronouns and anaphora ("He" → "Albert Einstein").
4. SPLIT compound claims ("X did A and B" → two separate claims).
5. PRESERVE exact numbers, dates, statistics, and quantities.
6. PRESERVE named entities exactly as stated.

SKIP (do NOT extract):
- Opinions, value judgements, subjective statements ("X is the best …").
- Hedged language ("might", "possibly", "is believed to").
- Questions, instructions, greetings.
- Trivially true statements ("water is wet").
- Future predictions.

CLAIM TYPES
- entity_fact: factual attribute of a named entity.
- numerical: involves a specific number or measurement.
- temporal: involves a date, year, or time reference.
- causal: describes cause and effect.
- relational: relationship between two or more entities.

OUTPUT — strict JSON array, nothing else:
[
  {
    "claim": "<self-contained factual statement>",
    "source_sentence": "<exact original sentence>",
    "claim_type": "<entity_fact|numerical|temporal|causal|relational>"
  }
]
"""

_HEDGE_WORDS = frozenset([
    "might", "maybe", "possibly", "perhaps", "probably",
    "i think", "it seems", "allegedly", "reportedly",
    "could be", "may be", "it is believed",
])


class ClaimDecomposer:
    """Extract atomic factual claims from text."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._nlp = spacy.load("en_core_web_sm")
        logger.info("ClaimDecomposer initialised")

    # ------------------------------------------------------------------
    @timed
    def decompose(self, text: str) -> List[Claim]:
        """Primary entry point — LLM extraction with spaCy fallback."""
        try:
            claims = self._llm_decompose(text)
            if claims:
                logger.info(f"LLM decomposed text into {len(claims)} claims")
                return claims
        except Exception as exc:
            logger.warning(f"LLM decomposition failed ({exc}); falling back to spaCy")

        claims = self._fallback_decompose(text)
        logger.info(f"Fallback decomposed text into {len(claims)} claims")
        return claims

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------
    def _llm_decompose(self, text: str) -> List[Claim]:
        raw = self.llm.generate(
            system=_SYSTEM_PROMPT,
            user=f"Extract all factual claims from this text:\n\n{text}",
            temperature=0.0,
            max_tokens=2048,
        )
        parsed = self._parse_json(raw)
        claims: List[Claim] = []
        for idx, item in enumerate(parsed):
            start, end = self._locate_span(text, item.get("source_sentence", ""))
            claims.append(Claim(
                id=f"c-{idx}",
                text=item["claim"],
                source_sentence=item.get("source_sentence", ""),
                claim_type=item.get("claim_type", "entity_fact"),
                char_start=start,
                char_end=end,
            ))
        return claims

    # ------------------------------------------------------------------
    # Fallback: spaCy sentence segmentation + heuristic filter
    # ------------------------------------------------------------------
    def _fallback_decompose(self, text: str) -> List[Claim]:
        doc = self._nlp(text)
        claims: List[Claim] = []
        for idx, sent in enumerate(doc.sents):
            s = sent.text.strip()
            if self._should_skip(s):
                continue
            claims.append(Claim(
                id=f"c-{idx}",
                text=s,
                source_sentence=s,
                claim_type="entity_fact",
                char_start=sent.start_char,
                char_end=sent.end_char,
            ))
        return claims

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(raw: str) -> list:
        """Robustly extract a JSON array from LLM output."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try extracting JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\[.*?])\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Last resort: find first [ … ]
        match = re.search(r"\[.*]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Could not parse JSON array from LLM response")

    @staticmethod
    def _should_skip(sentence: str) -> bool:
        lower = sentence.lower()
        if sentence.endswith("?"):
            return True
        if len(sentence.split()) < 5:
            return True
        return any(hw in lower for hw in _HEDGE_WORDS)

    @staticmethod
    def _locate_span(original: str, sentence: str) -> tuple[int, int]:
        if not sentence:
            return (-1, -1)
        idx = original.find(sentence)
        if idx != -1:
            return (idx, idx + len(sentence))
        # Fuzzy: match first few words
        words = sentence.split()[:6]
        partial = " ".join(words)
        idx = original.find(partial)
        if idx != -1:
            end = original.find(".", idx)
            return (idx, end + 1 if end > idx else idx + len(sentence))
        return (-1, -1)
