"""Short per-session chat history for the RAG prompt.

The widget does not send turns. HR keeps the last few Q&A pairs keyed by
conversation_id. Source footers and long answers are stripped so Sarvam is
not asked to reason over citations.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Deque, Sequence

_SOURCES_SPLIT = re.compile(r"\nSources[^\n]*:\n", re.IGNORECASE)


def strip_answer_for_history(text: str) -> str:
    body = _SOURCES_SPLIT.split(text or "", maxsplit=1)[0]
    return re.sub(r"\s+", " ", body).strip()


def is_welcome_turn(question: str, answer: str) -> bool:
    """Canned hi/hello replies must not occupy the next RAG prompt."""
    from app.services.greetings import WELCOME_MESSAGE, is_greeting

    if is_greeting(question):
        return True
    body = strip_answer_for_history(answer)
    if not body:
        return False
    return body == WELCOME_MESSAGE or body.startswith("Hi! I'm VedicMarriage Bot")


def format_history_text(
    turns: Sequence[tuple[str, str]],
    *,
    max_turns: int,
    max_chars: int,
) -> str:
    """Format the last user/assistant pairs. Oldest first."""
    lines: list[str] = []
    kept = [turn for turn in turns if not is_welcome_turn(turn[0], turn[1])]
    for question, answer in kept[-max_turns:]:
        q = strip_answer_for_history(question)[:max_chars]
        a = strip_answer_for_history(answer)[:max_chars]
        if not q and not a:
            continue
        if q:
            lines.append(f"User: {q}")
        if a:
            lines.append(f"Assistant: {a}")
    return "\n".join(lines)


class SessionHistory:
    """Last N Q&A turns per conversation_id. Process-local, not Mongo."""

    def __init__(self, max_turns: int = 3):
        self.max_turns = max(1, max_turns)
        self._turns: dict[str, Deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=self.max_turns)
        )

    def text(self, session_id: str, max_chars: int) -> str:
        return format_history_text(
            self._turns.get(session_id) or (),
            max_turns=self.max_turns,
            max_chars=max_chars,
        )

    def add(self, session_id: str, question: str, answer: str) -> None:
        q = strip_answer_for_history(question)
        a = strip_answer_for_history(answer)
        if not q and not a:
            return
        if is_welcome_turn(q, a):
            return
        self._turns[session_id].append((q, a))

    def replace_last_answer(self, session_id: str, answer: str) -> None:
        turns = self._turns.get(session_id)
        if not turns:
            return
        question, _ = turns[-1]
        turns[-1] = (question, strip_answer_for_history(answer))
