"""Generic answer grounding: block numeric claims not present in retrieved context.

No domain hardcoding — any invented count/price/%/scale figure is refused if it
does not appear (normalized) in the evidence text.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document

DEFAULT_REFUSAL = "I don't have that information in our HR documents."

# Digits / currency / % / magnitude words after a number, or word-number + scale.
_NUMERIC_CLAIM_RE = re.compile(
    r"""
    (?:[$₹€£]\s*)?
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,000 or 10,000,000
        |\d+(?:\.\d+)?                 # 10 or 3.5
    )
    (?:\s*(?:%|percent|million|billion|trillion|thousand|lakh|crore)\b)?
    |
    (?:
        one|two|three|four|five|six|seven|eight|nine|ten|
        eleven|twelve|thirteen|fourteen|fifteen|sixteen|
        seventeen|eighteen|nineteen|twenty|thirty|forty|
        fifty|sixty|seventy|eighty|ninety|hundred|thousand
    )
    \s+(?:million|billion|trillion|lakh|crore)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Skip tiny structural numbers that often appear as list indexes in answers.
_SKIP_IF_ONLY = re.compile(r"^\d{1,2}$")


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_numeric_claims(text: str) -> List[str]:
    """Surface forms of numeric claims in an answer (generic, not domain-specific)."""
    claims: List[str] = []
    seen: set[str] = set()
    for match in _NUMERIC_CLAIM_RE.finditer(text or ""):
        raw = match.group(0).strip()
        if not raw:
            continue
        # Keep single/double-digit alone only if tied to a unit/scale in the match
        digits_only = re.sub(r"[^\d.]", "", raw)
        has_scale = bool(
            re.search(
                r"%|percent|million|billion|trillion|thousand|lakh|crore|[$₹€£]",
                raw,
                re.I,
            )
        )
        if _SKIP_IF_ONLY.match(digits_only or "") and not has_scale:
            # Still allow multi-context: "60" alone is 2 digits — keep those.
            if len(digits_only.split(".")[0]) < 2:
                continue
        key = _normalize(raw)
        if key in seen:
            continue
        seen.add(key)
        claims.append(raw)
    return claims


def claim_supported_by_context(claim: str, context: str) -> bool:
    """True if the claim's normalized form (or digit core) appears in context."""
    ctx = _normalize(context)
    if not ctx:
        return False
    norm = _normalize(claim)
    if norm and norm in ctx:
        return True
    digits = re.sub(r"[^\d.]", "", claim)
    if digits and len(digits.replace(".", "")) >= 2 and digits in ctx.replace(" ", ""):
        # Require digit sequence in context; for scaled claims also require scale word
        scale = re.search(
            r"million|billion|trillion|thousand|lakh|crore|percent|%",
            claim,
            re.I,
        )
        if scale:
            return scale.group(0).lower().rstrip("%") in ctx or "%" in claim and "%" in ctx
        return True
    return False


def ungrounded_numeric_claims(
    answer: str, context: str
) -> List[str]:
    """Claims in the answer that are not supported by context text."""
    return [
        c
        for c in extract_numeric_claims(answer)
        if not claim_supported_by_context(c, context)
    ]


def context_from_documents(docs: Sequence[Document]) -> str:
    return "\n".join(d.page_content or "" for d in docs)


def enforce_numeric_grounding(
    answer: str,
    docs: Sequence[Document] | None = None,
    context: str | None = None,
    refusal: str = DEFAULT_REFUSAL,
) -> Tuple[str, List[str]]:
    """
    If the answer asserts numbers not present in retrieved evidence, replace
    with a generic refusal.

    Returns (possibly_replaced_answer, list_of_ungrounded_claims).
    """
    evidence = context if context is not None else context_from_documents(docs or [])
    bad = ungrounded_numeric_claims(answer, evidence)
    if bad:
        return refusal, bad
    return answer, []
