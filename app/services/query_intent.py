"""Generic query intent helpers for retrieval (no doc-id hardcoding)."""

import re

# Policy / legal style questions (refund, terms, privacy, etc.)
POLICY_QUERY = re.compile(
    r"\b("
    r"refund|cancellation|cancel(?:lation)?|auto[- ]?renew(?:al)?|"
    r"policy|policies|terms|conditions|agreement|privacy|legal|"
    r"dispute|mediation|liability|jurisdiction|force\s+majeure|"
    r"subscription\s+fee|non[- ]?refundable|deactivate|delete\s+(?:my\s+)?account|"
    r"grievance|dpa|dpdpa|data\s+fiduciary|unused\s+membership|"
    r"personal\s+data|data\s+protection|governing\s+laws?|applicable\s+laws?|"
    r"indemnif\w*|intellectual\s+property|termination|obligation|"
    r"consent|fiduciary|cookie|trademark|copyright"
    r")\b",
    re.IGNORECASE,
)

# Refund-specific: dense often ranks gold earlier than BM25 on this corpus
REFUND_QUERY = re.compile(
    r"\b(refund|non[- ]?refundable|duplicate\s+payment|money\s+back|"
    r"cancel(?:lation)?(?:\s+policy)?|force\s+majeure)\b",
    re.IGNORECASE,
)

REGISTRATION_QUERY = re.compile(
    r"\b(register|registration|sign\s*up|signup|join|create\s+(?:an?\s+)?account|new\s+account)\b",
    re.IGNORECASE,
)


def is_policy_query(query: str) -> bool:
    """True when the question looks like policy/legal, not FAQ onboarding."""
    return bool(POLICY_QUERY.search(query or ""))


def is_refund_query(query: str) -> bool:
    return bool(REFUND_QUERY.search(query or ""))


def is_registration_query(query: str) -> bool:
    return bool(REGISTRATION_QUERY.search(query or ""))


def should_use_hybrid_retrieval(query: str, policy_only: bool) -> bool:
    """
    Decide whether BM25 fusion should run for this query.

    - policy_only=False → hybrid for all queries
    - policy_only=True  → hybrid for policy/terms-like queries, but not refund
      (refund MRR is better with dense-only on current eval)
    """
    if not policy_only:
        return True
    if is_refund_query(query):
        return False
    return is_policy_query(query)
