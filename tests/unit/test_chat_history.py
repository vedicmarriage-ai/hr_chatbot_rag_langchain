from app.services.chat_history import (
    SessionHistory,
    format_history_text,
    strip_answer_for_history,
)
from app.services.greetings import WELCOME_MESSAGE


def test_strip_drops_sources_footer():
    text = 'Registration is FREE.\n\nSources (retrieved excerpts — not proof):\n- FAQ.docx: "free"'
    assert strip_answer_for_history(text) == "Registration is FREE."


def test_format_keeps_last_three_turns_only():
    turns = [(f"q{i}", f"a{i}") for i in range(5)]
    text = format_history_text(turns, max_turns=3, max_chars=80)
    assert "q0" not in text and "q1" not in text
    assert "User: q2" in text
    assert "Assistant: a4" in text
    assert text.count("User:") == 3


def test_welcome_turns_are_not_stored_or_formatted():
    store = SessionHistory(max_turns=3)
    store.add("s", "hi", WELCOME_MESSAGE)
    store.add("s", "Is registration free?", "It is FREE.")
    text = store.text("s", 200)
    assert "not signed in" not in text
    assert "User: hi" not in text
    assert "FREE" in text
    formatted = format_history_text(
        [("hello", WELCOME_MESSAGE), ("Refunds?", "Non-refundable.")],
        max_turns=3,
        max_chars=200,
    )
    assert "VedicMarriage Bot" not in formatted
    assert "Refunds?" in formatted


def test_session_history_is_isolated():
    store = SessionHistory(max_turns=3)
    store.add("a", "Is registration free?", "It is FREE.")
    store.add("b", "Refund?", "Non-refundable.")
    assert "FREE" in store.text("a", 80)
    assert "Refund" not in store.text("a", 80)
    assert "FREE" not in store.text("b", 80)
