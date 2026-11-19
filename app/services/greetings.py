"""Canned openers. Do not send greetings through retrieval or Sarvam."""

from __future__ import annotations

import re

# Keep in sync with the design-library `VDM_CHATBOT_WELCOME_GUEST` copy.
WELCOME_MESSAGE = (
    "Hi! I'm VedicMarriage Bot. It looks like you're not signed in yet. "
    "Signing in will let me tailor support to your membership and account as we chat. "
    "Ask me about registration, membership, or refunds."
)

_GREETING_ONLY = re.compile(
    r"^\s*("
    r"hi+|hii+|hello+|hey+|heya|howdy|yo|"
    r"namaste|namaskar|"
    r"good\s+(morning|afternoon|evening)|"
    r"greetings|hola|hai"
    r")"
    r"(\s+(there|bot|hi|hello|hey))*"
    r"\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or len(stripped) > 80:
        return False
    return bool(_GREETING_ONLY.match(stripped))
