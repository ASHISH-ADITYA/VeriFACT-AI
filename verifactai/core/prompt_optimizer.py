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
# Domain detection & keyword maps
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "programming": [
        "python", "java", "javascript", "code", "function", "api", "bug",
        "debug", "algorithm", "database", "sql", "html", "css", "react",
        "class", "method", "library", "framework", "git", "deploy", "docker",
        "kubernetes", "rust", "golang", "typescript", "node", "backend",
        "frontend", "sorting", "recursion", "regex", "http", "rest",
    ],
    "data_science": [
        "data", "machine learning", "ml", "model", "training", "dataset",
        "neural network", "deep learning", "pandas", "numpy", "sklearn",
        "regression", "classification", "clustering", "nlp", "llm",
        "transformer", "fine-tune", "accuracy", "precision", "recall",
        "statistics", "visualization", "matplotlib", "pytorch", "tensorflow",
    ],
    "business": [
        "marketing", "strategy", "revenue", "roi", "customer", "sales",
        "business", "market", "growth", "startup", "product", "pricing",
        "competition", "stakeholder", "kpi", "budget", "forecast",
    ],
    "science": [
        "research", "hypothesis", "experiment", "study", "evidence",
        "scientific", "biology", "chemistry", "physics", "clinical",
        "peer-reviewed", "methodology", "findings", "analysis",
    ],
    "writing": [
        "essay", "write", "article", "blog", "story", "email", "letter",
        "report", "summary", "creative writing", "copywriting", "tone",
        "draft", "outline", "narrative", "paragraph",
    ],
    "education": [
        "learn", "teach", "explain", "student", "course", "curriculum",
        "lesson", "tutorial", "beginner", "advanced", "concept", "exam",
    ],
}

_DOMAIN_CONTEXT: dict[str, str] = {
    "programming": (
        "Include working code examples with comments, note time/space complexity "
        "where relevant, mention edge cases, and highlight common pitfalls."
    ),
    "data_science": (
        "Include mathematical intuition, practical code snippets, evaluation "
        "metrics, and discuss trade-offs between approaches."
    ),
    "business": (
        "Ground advice in real-world examples, include relevant metrics or KPIs, "
        "and consider both short-term and long-term implications."
    ),
    "science": (
        "Cite established principles, distinguish between established consensus "
        "and emerging research, and note limitations of current evidence."
    ),
    "writing": (
        "Consider the target audience and purpose, suggest appropriate tone and "
        "structure, and provide concrete before/after examples."
    ),
    "education": (
        "Use clear analogies, build from fundamentals to advanced concepts, and "
        "include practice exercises or self-check questions."
    ),
}

_DOMAIN_ROLES: dict[str, str] = {
    "programming": "an expert software engineer with deep knowledge of best practices",
    "data_science": "a senior data scientist experienced in both theory and production ML",
    "business": "a seasoned business strategist with MBA-level analytical skills",
    "science": "a research scientist skilled at explaining complex findings clearly",
    "writing": "a professional editor and writing coach",
    "education": "an experienced educator who excels at making complex topics accessible",
}


def _detect_domain(text: str) -> str | None:
    """Return the best-matching domain key, or None."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lower)
        if count:
            scores[domain] = count
    if not scores:
        return None
    return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Prompt category detection & template application
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("tell_me", re.compile(
        r"^(tell me|talk|explain|describe|discuss|elaborate|give me info)\s+"
        r"(about|on|regarding)\s+(.+)",
        re.IGNORECASE,
    )),
    ("how_to", re.compile(
        r"^(how (do|can|should|would|to)|steps? (to|for)|guide (to|for|on))\s+(.+)",
        re.IGNORECASE,
    )),
    ("what_is", re.compile(
        r"^(what (is|are|was|were)|define|meaning of)\s+(.+)",
        re.IGNORECASE,
    )),
    ("compare", re.compile(
        r"^(compare|difference[s]? between|(.+)\s+vs\.?\s+(.+)|contrast)\s*(.+)",
        re.IGNORECASE,
    )),
    ("list", re.compile(
        r"^(list|give me|name|enumerate|what are (the|some))\s+(.+)",
        re.IGNORECASE,
    )),
    ("why", re.compile(
        r"^(why (is|are|do|does|did|was|were|should|would|can))\s+(.+)",
        re.IGNORECASE,
    )),
]


def _detect_category(text: str) -> tuple[str, re.Match[str] | None]:
    """Return (category_name, match) or ('general', None)."""
    stripped = text.strip().rstrip("?").strip()
    for name, pat in _CATEGORY_PATTERNS:
        m = pat.match(stripped)
        if m:
            return name, m
    return "general", None


def _extract_subject(text: str) -> str:
    """Pull the core subject from the prompt, stripping common prefixes."""
    s = text.strip().rstrip("?").strip()
    s = re.sub(
        r"^(tell me about|how (do|can|to|should)|what (is|are)|explain|describe|compare)\s+",
        "", s, flags=re.IGNORECASE,
    ).strip()
    return s or text.strip()


def _apply_template(category: str, subject: str, domain: str | None) -> str:
    """Return a fully rewritten prompt based on detected category and domain."""
    role_prefix = ""
    if domain and domain in _DOMAIN_ROLES:
        role_prefix = f"As {_DOMAIN_ROLES[domain]}, "

    domain_suffix = ""
    if domain and domain in _DOMAIN_CONTEXT:
        domain_suffix = f"\n\n{_DOMAIN_CONTEXT[domain]}"

    templates: dict[str, str] = {
        "tell_me": (
            f"{role_prefix}provide a comprehensive analysis of {subject}.\n\n"
            f"Cover the following aspects:\n"
            f"1. Core definition and key concepts\n"
            f"2. Historical context and evolution\n"
            f"3. How it works (mechanisms or principles)\n"
            f"4. Real-world applications and examples\n"
            f"5. Current trends and future outlook\n\n"
            f"Be specific, use concrete examples, and cite sources where possible."
            f"{domain_suffix}"
        ),
        "how_to": (
            f"{role_prefix}provide a step-by-step guide for {subject}.\n\n"
            f"For each step, include:\n"
            f"- **What to do**: Clear, actionable instruction\n"
            f"- **Why it matters**: The reasoning behind this step\n"
            f"- **Common mistakes**: Pitfalls to avoid\n"
            f"- **Expected outcome**: What success looks like\n\n"
            f"Include prerequisites, estimated time, and difficulty level. "
            f"End with troubleshooting tips for the most common issues."
            f"{domain_suffix}"
        ),
        "what_is": (
            f"{role_prefix}define {subject} precisely, then explain:\n\n"
            f"1. **Origin and history**: When and why it emerged\n"
            f"2. **How it works**: Core mechanisms or principles\n"
            f"3. **Significance**: Why it matters in its field\n"
            f"4. **Real-world applications**: Concrete use cases\n"
            f"5. **Alternatives and comparisons**: How it relates to similar concepts\n\n"
            f"Use clear language suitable for someone with basic domain knowledge."
            f"{domain_suffix}"
        ),
        "compare": (
            f"{role_prefix}compare {subject} across these dimensions:\n\n"
            f"| Dimension | Option A | Option B |\n"
            f"|-----------|----------|----------|\n"
            f"| Core purpose | | |\n"
            f"| Key strengths | | |\n"
            f"| Limitations | | |\n"
            f"| Best suited for | | |\n"
            f"| Cost/complexity | | |\n\n"
            f"After the comparison, provide a clear recommendation for different "
            f"scenarios and use cases. Be balanced and evidence-based."
            f"{domain_suffix}"
        ),
        "list": (
            f"{role_prefix}list and explain the key items related to {subject}.\n\n"
            f"For each item, provide:\n"
            f"- A concise name or title\n"
            f"- A 1-2 sentence explanation of what it is and why it matters\n"
            f"- A practical example or use case\n\n"
            f"Organize from most to least important. Limit to the top entries "
            f"unless completeness is essential."
            f"{domain_suffix}"
        ),
        "why": (
            f"{role_prefix}explain why {subject}.\n\n"
            f"Structure your answer as:\n"
            f"1. **Short answer**: A direct 1-2 sentence response\n"
            f"2. **Root causes**: The underlying factors driving this\n"
            f"3. **Evidence**: Data, studies, or examples that support the explanation\n"
            f"4. **Nuance**: Any caveats, exceptions, or alternative viewpoints\n\n"
            f"Think step by step and be precise."
            f"{domain_suffix}"
        ),
    }

    if category in templates:
        return templates[category]

    # General / fallback template
    return (
        f"{role_prefix}address the following request thoroughly:\n\n"
        f"{subject}\n\n"
        f"Structure your response with clear headings and provide:\n"
        f"- Specific, actionable information\n"
        f"- Concrete examples or evidence\n"
        f"- Key takeaways or summary\n\n"
        f"Be precise and focus on practical value."
        f"{domain_suffix}"
    )


# ---------------------------------------------------------------------------
# Structural enhancement helpers
# ---------------------------------------------------------------------------

def _has_role(text: str) -> bool:
    """Check if prompt already assigns a role/persona."""
    return bool(re.search(
        r"\b(as an?|act as|you are|pretend|imagine you'?re|role)\b",
        text, re.IGNORECASE,
    ))


def _has_format_request(text: str) -> bool:
    """Check if prompt already specifies an output format."""
    return bool(re.search(
        r"\b(markdown|json|table|bullet|numbered|list|format|headers?|csv|xml)\b",
        text, re.IGNORECASE,
    ))


def _has_specificity(text: str) -> bool:
    """Check if prompt contains specific constraints or parameters."""
    indicators = [
        r"\d+",  # numbers
        r"\b(exactly|at least|at most|maximum|minimum|between|within)\b",
        r"\b(must|should|require|constraint|limit|restrict)\b",
        r"\b(example|instance|such as|e\.g\.|i\.e\.)\b",
    ]
    matches = sum(1 for pat in indicators if re.search(pat, text, re.IGNORECASE))
    return matches >= 2


def _has_step_by_step(text: str) -> bool:
    return bool(re.search(
        r"\b(step[- ]by[- ]step|chain of thought|think through|reason through)\b",
        text, re.IGNORECASE,
    ))


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


def _score_prompt(text: str) -> tuple[int, list[str]]:
    """Score the original prompt 0-100 and collect improvement notes.

    Returns (score, improvements) where improvements has at most 5 items.
    """
    score = 35  # start low; earn points for quality signals
    improvements: list[str] = []
    word_count = len(text.split())

    # --- Length ---
    if word_count < 5:
        improvements.append(
            "Prompt is extremely short — add detail about your goal, "
            "context, and desired output format."
        )
    elif word_count < 12:
        score += 5
        improvements.append(
            "Add more context: specify the audience, purpose, and scope "
            "to get a tailored response."
        )
    elif word_count < 30:
        score += 12
    else:
        score += 18

    # --- Vague words ---
    vague = _has_vague_words(text)
    if vague:
        unique = sorted(set(w.lower() for w in vague))
        improvements.append(
            f"Replace vague terms ({', '.join(repr(w) for w in unique[:3])}) "
            "with specific, concrete language."
        )
        score -= 3 * min(len(unique), 4)
    else:
        score += 8

    # --- Yes/no question ---
    if _is_yes_no_question(text):
        improvements.append(
            "Rephrase the yes/no question as an open-ended query to get "
            "a richer, more informative answer."
        )
        score -= 8

    # --- Role / persona ---
    if _has_role(text):
        score += 12
    elif word_count >= 8:
        improvements.append(
            "Assign an expert role (e.g. 'As a senior engineer...') to "
            "ground the response in domain expertise."
        )

    # --- Format request ---
    if _has_format_request(text):
        score += 10
    elif word_count >= 8:
        improvements.append(
            "Specify your desired output format (markdown, table, bullet "
            "list, etc.) for a better-structured response."
        )

    # --- Specificity ---
    if _has_specificity(text):
        score += 12

    # --- Step-by-step ---
    if _has_step_by_step(text):
        score += 5

    # --- Structure (multi-line, lists) ---
    if "\n" in text.strip() or re.search(r"[-*\d]+[.)]\s", text):
        score += 8

    # --- Context cues ---
    if not _lacks_context(text):
        score += 8

    score = max(0, min(100, score))

    # Trim to at most 5 improvements
    return score, improvements[:5]


# ---------------------------------------------------------------------------
# Rule-based optimizer (no LLM required, <100 ms)
# ---------------------------------------------------------------------------


def _rule_based_optimize(prompt: str) -> OptimizationResult:
    """Transform the prompt using professional prompt-engineering patterns.

    Detects category, domain, and quality signals, then produces a complete,
    ready-to-use rewritten prompt.
    """
    text = prompt.strip()
    word_count = len(text.split())

    # --- Score the original ---
    score, improvements = _score_prompt(text)

    # --- If the prompt is already high quality, return with minor tweaks ---
    if score >= 90:
        return OptimizationResult(
            original=prompt,
            suggested=prompt,
            improvements=["Prompt is already well-structured and specific."],
            score=score,
        )

    # --- Detect domain and category ---
    domain = _detect_domain(text)
    category, _match = _detect_category(text)
    subject = _extract_subject(text)

    # --- Handle yes/no rewrites ---
    if _is_yes_no_question(text):
        text = re.sub(
            r"^(Is|Are|Was|Were|Do|Does|Did|Can|Could|Will|Would|Should|Has|Have|Had)\b",
            "Explain whether",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        if text.endswith("?"):
            text = text[:-1] + "."
        subject = _extract_subject(text)

    # --- For very short or clearly categorized prompts, use templates ---
    if word_count < 30 or category != "general":
        suggested = _apply_template(category, subject, domain)
    else:
        # Longer prompts: enhance in-place rather than replacing
        parts: list[str] = []

        # Add role if missing
        if not _has_role(text) and domain and domain in _DOMAIN_ROLES:
            parts.append(f"As {_DOMAIN_ROLES[domain]},")

        parts.append(text.rstrip(".") + ".")

        # Add analytical directive for complex queries
        if not _has_step_by_step(text) and word_count > 15:
            parts.append("\nThink step by step.")

        # Add format if missing
        if not _has_format_request(text):
            parts.append(
                "\nUse markdown with clear headers. Include a brief summary "
                "at the end highlighting the key takeaways."
            )

        # Add domain context
        if domain and domain in _DOMAIN_CONTEXT:
            parts.append(f"\n{_DOMAIN_CONTEXT[domain]}")

        # Add specificity constraint
        if not _has_specificity(text):
            parts.append(
                "\nFocus on the most important points and support each with "
                "a concrete example or data point."
            )

        suggested = " ".join(parts)

    # Capitalize first letter
    if suggested and suggested[0].islower():
        suggested = suggested[0].upper() + suggested[1:]

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
