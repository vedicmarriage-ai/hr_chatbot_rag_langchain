"""HR generation. The gateway and local model are mocked."""

from types import SimpleNamespace

from app.services import answer_generation
from app.services.sarvam_gateway import GatewayAnswer, SarvamGatewayError


class _Prompt:
    def format(self, **kwargs):
        return "Context:\n{context}\nQuestion: {question}".format(**kwargs)


PROMPT = _Prompt()
DOC = SimpleNamespace(page_content="Annual leave is 18 days.")
REFUSAL = "I don't have that information in our HR documents."


def test_remote_answer_does_not_call_local(monkeypatch):
    monkeypatch.setattr(
        answer_generation,
        "complete",
        lambda _prompt: GatewayAnswer("18 days.", 12.5, 100, 40, 140),
    )
    called = {"local": False}

    def local(_prompt):
        called["local"] = True
        return "local"

    record = answer_generation.generate_from_docs(
        question="How much leave?",
        docs=[DOC],
        prompt_template=PROMPT,
        chat_history="",
        refusal=REFUSAL,
        local_generate=local,
        fallback_enabled=True,
    )
    assert record.text == "18 days."
    assert record.provider == "sarvam"
    assert record.prompt_tokens == 100
    assert record.total_tokens == 140
    assert called["local"] is False


def test_503_timeout_and_connection_use_local(monkeypatch):
    def fail(_prompt):
        raise SarvamGatewayError("status 503")

    monkeypatch.setattr(answer_generation, "complete", fail)
    seen = {}

    def local(prompt):
        seen["prompt"] = prompt
        return "local leave answer"

    record = answer_generation.generate_from_docs(
        question="How much leave?",
        docs=[DOC],
        prompt_template=PROMPT,
        chat_history="",
        refusal=REFUSAL,
        local_generate=local,
        fallback_enabled=True,
    )
    assert "Annual leave is 18 days." in seen["prompt"]
    assert record.provider == "ollama"
    assert record.text == "local leave answer"
    assert record.prompt_tokens is None
    assert record.latency_ms >= 0


def test_empty_retrieval_calls_neither_model(monkeypatch):
    def boom(_prompt):
        raise AssertionError("gateway called")

    monkeypatch.setattr(answer_generation, "complete", boom)

    def local(_prompt):
        raise AssertionError("local called")

    record = answer_generation.generate_from_docs(
        question="Salary band?",
        docs=[],
        prompt_template=PROMPT,
        chat_history="",
        refusal=REFUSAL,
        local_generate=local,
        fallback_enabled=True,
    )
    assert record.text == REFUSAL
    assert record.provider == "none"
