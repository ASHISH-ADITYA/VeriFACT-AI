"""
Prompt Optimizer — suggests improved versions of user prompts in real-time.

Two execution paths:
  1. LLM-powered (provider != "none"): uses the configured LLM to rewrite the
     prompt following top-tier prompt-engineering best practices.
  2. Rule-based (provider == "none" / HF Space free tier): fast heuristic
     analysis that flags common weaknesses and suggests concrete fixes.

Both paths return a uniform result dict:
  {"original", "suggested", "improvements", "score"}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from utils.helpers import logger

if TYPE_CHECKING:
    from core.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    original: str
    suggested: str
    improvements: list[str] = field(default_factory=list)
    score: int = 50  # 0-100

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "suggested": self.suggested,
            "improvements": self.improvements,
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

_VAGUE_WORDS = re.compile(
    r"\b(stuff|things?|some|a lot|somehow|something|anything|whatever|etc)\b",
    re.IGNORECASE,
)

_YES_NO_STARTERS = re.compile(
    r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b",
    re.IGNORECASE,
)


def _is_too_short(text: str) -> bool:
    return len(text.strip()) < 20


def _has_vague_words(text: str) -> list[str]:
    return _VAGUE_WORDS.findall(text)


def _is_yes_no_question(text: str) -> bool:
    stripped = text.strip().rstrip("?").strip()
    return bool(_YES_NO_STARTERS.match(stripped)) and text.strip().endswith("?")


def _lacks_structure(text: str) -> bool:
    """True when the prompt is a single long blob with no list markers or newlines."""
    if "\n" in text.strip():
        return False
    if re.search(r"^\s*[-*\d]+[.)]\s", text, re.MULTILINE):
        return False
    return len(text.split()) > 25


def _lacks_context(text: str) -> bool:
    """Heuristic: prompt has no contextual framing (no 'because', 'for', 'about', etc.)."""
    context_cues = re.compile(
        r"\b(because|context|background|for|about|regarding|related to|in the context of)\b",
        re.IGNORECASE,
    )
    return not context_cues.search(text) and len(text.split()) < 15


# ---------------------------------------------------------------------------
# Rule-based optimizer (no LLM required, <100 ms)
# ---------------------------------------------------------------------------


def _rule_based_optimize(prompt: str) -> OptimizationResult:
    improvements: list[str] = []
    score = 60  # decent baseline

    # 1. Too short
    if _is_too_short(prompt):
        improvements.append(
            "Prompt is very short — add more detail about what you need, "
            "including context, constraints, and desired output format."
        )
        score -= 15

    # 2. Vague words
    vague = _has_vague_words(prompt)
    if vague:
        unique = sorted(set(w.lower() for w in vague))
        improvements.append(
            f"Replace vague words ({', '.join(repr(w) for w in unique)}) "
            "with specific, concrete terms."
        )
        score -= 5 * min(len(unique), 4)

    # 3. Yes/no question
    if _is_yes_no_question(prompt):
        improvements.append(
            "Rephrase the yes/no question as an open-ended query "
            "(e.g., 'Explain how…' or 'What are the factors that…') "
            "to get a more informative answer."
        )
        score -= 10

    # 4. Lacks context
    if _lacks_context(prompt):
        improvements.append(
            "Add context: specify the domain, audience, purpose, or background "
            "so the model can tailor its response."
        )
        score -= 10

    # 5. No structure
    if _lacks_structure(prompt):
        improvements.append(
            "Break the prompt into structured sections using bullet points or "
            "numbered steps for clarity."
        )
        score -= 5

    # Build suggested prompt
    if improvements:
        suggested = prompt.rstrip()
        addenda: list[str] = []
        if _is_too_short(prompt):
            addenda.append("Please provide a detailed explanation including relevant context.")
        if _is_yes_no_question(prompt):
            # Rewrite opener
            suggested = re.sub(
                r"^(Is|Are|Was|Were|Do|Does|Did|Can|Could|Will|Would|Should|Has|Have|Had)\b",
                "Explain whether",
                suggested,
                count=1,
                flags=re.IGNORECASE,
            )
            if suggested.endswith("?"):
                suggested = suggested[:-1] + "."
        if _lacks_context(prompt):
            addenda.append("Include the relevant context, domain, and purpose.")
        if _lacks_structure(prompt):
            addenda.append("Organize the request as a numbered list of requirements.")
        if addenda:
            suggested = suggested.rstrip(".") + ". " + " ".join(addenda)
    else:
        suggested = prompt
        score = 85  # already good

    score = max(0, min(100, score))

    return OptimizationResult(
        original=prompt,
        suggested=suggested,
        improvements=improvements,
        score=score,
    )


# ---------------------------------------------------------------------------
# LLM-powered optimizer
# ---------------------------------------------------------------------------

_OPTIMIZER_SYSTEM_PROMPT = """\
You are a world-class prompt engineer. Your task is to take a user's raw prompt \
and rewrite it into an optimised version that follows elite prompt-engineering \
best practices:

1. **Be specific and detailed** — replace vague requests with precise \
   instructions.
2. **Provide context and constraints** — state the domain, audience, and any \
   boundaries.
3. **Use structured output requests** — ask for numbered lists, tables, or \
   headings when appropriate.
4. **Include examples where helpful** — show the model what good output looks \
   like.
5. **Specify a role or persona** — e.g., "Act as a senior data scientist…"
6. **Set quality expectations** — mention depth, length, tone, and format.

Return your answer as valid JSON with exactly these keys:
{
  "suggested": "<the improved prompt>",
  "improvements": ["<improvement 1>", "<improvement 2>", ...],
  "score": <integer 0-100 rating the ORIGINAL prompt quality>
}

Return ONLY the JSON object, no markdown fences, no extra text.\
"""


def _llm_optimize(prompt: str, llm: LLMClient) -> OptimizationResult:
    user_msg = f"Original prompt to optimise:\n\n{prompt}"

    raw = llm.generate(
        user=user_msg,
        system=_OPTIMIZER_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=1024,
    )

    if raw is None:
        logger.warning("LLM returned None for prompt optimisation; falling back to rules.")
        return _rule_based_optimize(prompt)

    # Strip markdown fences if the model wrapped them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON for prompt optimisation; falling back to rules.")
        return _rule_based_optimize(prompt)

    suggested = str(data.get("suggested", prompt))
    improvements = list(data.get("improvements", []))
    score = int(data.get("score", 50))
    score = max(0, min(100, score))

    return OptimizationResult(
        original=prompt,
        suggested=suggested,
        improvements=improvements,
        score=score,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PromptOptimizer:
    """Suggest improved versions of user prompts.

    Parameters
    ----------
    llm_client:
        An initialised ``LLMClient`` instance (may be ``None``).
    provider:
        The LLM provider string from config (e.g. ``"ollama"``, ``"none"``).
    """

    def __init__(self, llm_client: LLMClient | None = None, provider: str = "none") -> None:
        self._llm = llm_client
        self._provider = provider

    @property
    def has_llm(self) -> bool:
        return self._provider != "none" and self._llm is not None

    def optimize(self, prompt: str) -> OptimizationResult:
        """Return an ``OptimizationResult`` for the given prompt.

        Uses the LLM path when available, otherwise falls back to fast
        rule-based heuristics (<100 ms).
        """
        if not prompt or not prompt.strip():
            return OptimizationResult(
                original=prompt,
                suggested=prompt,
                improvements=["Prompt is empty — please provide a question or instruction."],
                score=0,
            )

        if self.has_llm:
            return _llm_optimize(prompt, self._llm)  # type: ignore[arg-type]
        return _rule_based_optimize(prompt)
