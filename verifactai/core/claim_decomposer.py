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
from typing import TYPE_CHECKING

import spacy

from utils.helpers import logger, timed

if TYPE_CHECKING:
    from core.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A single atomic factual claim extracted from LLM output."""

    id: str  # stable identifier e.g. "c-0", "c-1"
    text: str  # self-contained factual statement
    source_sentence: str  # original sentence it came from
    claim_type: str = "entity_fact"  # entity_fact|numerical|temporal|causal|relational
    char_start: int = -1  # start offset in original text
    char_end: int = -1  # end offset in original text
    # Filled downstream
    evidence: list = field(default_factory=list)
    verdict: str | None = None
    confidence: float | None = None
    best_evidence: object | None = None
    nli_scores: dict | None = None
    all_nli_results: list | None = None
    uncertainty: float | None = None
    stability: float | None = None
    correction: str | None = None


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

_HEDGE_WORDS = frozenset(
    [
        "might",
        "maybe",
        "possibly",
        "perhaps",
        "probably",
        "i think",
        "it seems",
        "allegedly",
        "reportedly",
        "could be",
        "may be",
        "it is believed",
    ]
)


class ClaimDecomposer:
    """Extract atomic factual claims from text."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._nlp = spacy.load("en_core_web_sm")
        logger.info("ClaimDecomposer initialised")

    # ------------------------------------------------------------------
    @timed
    def decompose(self, text: str) -> list[Claim]:
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
    def _llm_decompose(self, text: str) -> list[Claim]:
        raw = self.llm.generate(
            system=_SYSTEM_PROMPT,
            user=f"Extract all factual claims from this text:\n\n{text}",
            temperature=0.0,
            max_tokens=2048,
        )
        parsed = self._parse_json(raw)
        claims: list[Claim] = []
        for idx, item in enumerate(parsed):
            start, end = self._locate_span(text, item.get("source_sentence", ""))
            claims.append(
                Claim(
                    id=f"c-{idx}",
                    text=item["claim"],
                    source_sentence=item.get("source_sentence", ""),
                    claim_type=item.get("claim_type", "entity_fact"),
                    char_start=start,
                    char_end=end,
                )
            )
        return claims

    # ------------------------------------------------------------------
    # Fallback: spaCy sentence segmentation + heuristic filter
    # ------------------------------------------------------------------
    def _fallback_decompose(self, text: str, max_claims: int = 20) -> list[Claim]:
        doc = self._nlp(text)
        candidates: list[tuple[float, Claim]] = []
        for idx, sent in enumerate(doc.sents):
            s = sent.text.strip()
            if self._should_skip(s):
                continue
            claim = Claim(
                id=f"c-{idx}",
                text=s,
                source_sentence=s,
                claim_type=self._infer_type(s),
                char_start=sent.start_char,
                char_end=sent.end_char,
            )
            # Rank by fact-density: sentences with numbers, dates, names score higher
            score = self._fact_density(s)
            candidates.append((score, claim))

        # Sort by fact-density descending, take top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        claims = [c for _, c in candidates[:max_claims]]
        # Re-sort by position in text for natural order
        claims.sort(key=lambda c: c.char_start)
        return claims

    @staticmethod
    def _fact_density(sentence: str) -> float:
        """Score how likely a sentence contains verifiable facts. Higher = more factual."""
        s = 1.0  # base score
        lower = sentence.lower()
        # Numbers, dates, measurements boost score
        if re.search(r"\d", sentence):
            s += 2.0
        # Named entities (capitalized words not at sentence start)
        caps = re.findall(r"(?<!^)(?<!\. )[A-Z][a-z]+", sentence)
        s += len(caps) * 0.5
        # Specific factual keywords
        fact_words = ["is", "was", "are", "were", "located", "founded", "invented", "discovered",
                      "born", "died", "built", "capital", "population", "contains", "consists"]
        s += sum(0.5 for w in fact_words if w in lower)
        # Longer sentences with more content
        s += min(len(sentence.split()) / 20.0, 1.0)
        return s

    @staticmethod
    def _infer_type(sentence: str) -> str:
        """Infer claim type from sentence content."""
        lower = sentence.lower()
        if re.search(r"\b\d{4}\b|century|year|era|period|age", lower):
            return "temporal"
        if re.search(r"\b\d+\.?\d*\s*(percent|%|km|miles|meters|kg|pounds|degrees|million|billion)", lower):
            return "numerical"
        if any(w in lower for w in ["cause", "because", "result", "lead to", "due to"]):
            return "causal"
        return "entity_fact"

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
        """Aggressively filter non-verifiable sentences.

        Only keep sentences that contain concrete, checkable facts.
        Skip opinions, explanations, meta-text, common sense, and filler.
        """
        lower = sentence.lower().strip()
        words = lower.split()

        # Too short or too long
        if len(words) < 5 or len(words) > 60:
            return True

        # Questions
        if sentence.rstrip().endswith("?"):
            return True

        # Hedged language
        if any(hw in lower for hw in _HEDGE_WORDS):
            return True

        # Meta-commentary, instructions, transitions, filler
        skip_starts = [
            "here are", "here is", "here's", "if you", "let me", "i hope",
            "are you", "do you", "would you", "feel free", "please",
            "note:", "tip:", "warning:", "important:", "remember",
            "in summary", "to summarize", "in conclusion", "overall",
            "for example", "for instance", "in other words", "that is",
            "this means", "this is because", "this is why", "this is how",
            "as a result", "as mentioned", "as noted", "as stated",
            "it is worth", "it's worth", "it should be noted",
            "the reason", "one reason", "another reason",
            "first,", "second,", "third,", "finally,", "additionally,",
            "furthermore,", "moreover,", "however,", "therefore,",
            "in fact,", "indeed,", "specifically,", "essentially,",
            "basically,", "simply put", "put simply",
        ]
        if any(lower.startswith(s) for s in skip_starts):
            return True

        # Numbered/bulleted list headers
        if re.match(r"^\d+[\.\)]\s*\w+(\s+\w+)?$", sentence.strip()):
            return True

        # Subjective/opinion indicators
        opinion_words = [
            "best", "worst", "amazing", "terrible", "beautiful", "ugly",
            "should", "must", "need to", "have to", "ought to",
            "i think", "i believe", "in my opinion", "personally",
            "interesting", "fascinating", "remarkable", "significant",
            "it's important", "it is important", "crucial", "essential",
            "obviously", "clearly", "certainly", "definitely",
        ]
        if any(ow in lower for ow in opinion_words):
            return True

        # Common sense / tautologies (not worth verifying)
        trivial = [
            "this is a", "this was a", "it is a", "it was a",
            "there are many", "there are several", "there are various",
            "people often", "many people", "some people",
        ]
        if any(lower.startswith(t) for t in trivial):
            return True

        # Must contain at least one concrete signal to be verifiable:
        # numbers, proper nouns (capitalized words), dates, measurements
        has_number = bool(re.search(r"\d", sentence))
        has_proper_noun = bool(re.search(r"[A-Z][a-z]{2,}", sentence[1:]))  # skip first char
        has_factual_verb = any(v in lower for v in [
            " is ", " was ", " are ", " were ", " has ", " had ",
            " located ", " founded ", " invented ", " discovered ",
            " born ", " died ", " built ", " created ", " published ",
            " won ", " lost ", " defeated ", " signed ", " established ",
        ])

        return not (has_number or has_proper_noun or has_factual_verb)

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
