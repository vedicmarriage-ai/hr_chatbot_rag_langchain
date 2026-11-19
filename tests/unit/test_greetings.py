from app.services.greetings import WELCOME_MESSAGE, is_greeting


def test_plain_greetings():
    for text in ("hi", "Hi!", "hello", "Hey there", "namaste", "good morning"):
        assert is_greeting(text), text


def test_real_questions_are_not_greetings():
    for text in (
        "hi, is registration free?",
        "hello what is the refund policy",
        "Is registration free?",
    ):
        assert not is_greeting(text), text


def test_welcome_mentions_vedicmarriage_bot():
    assert "VedicMarriage Bot" in WELCOME_MESSAGE
    assert "not signed in" in WELCOME_MESSAGE.lower()
