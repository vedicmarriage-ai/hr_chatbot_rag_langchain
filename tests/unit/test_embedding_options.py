from app.services.embedding_service import EmbeddingService, EmbeddingType


def test_ollama_embeddings_keep_alive_without_oversubscribed_threads(monkeypatch):
    captured = {}

    class FakeOllamaEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "app.services.embedding_service.OllamaEmbeddings",
        FakeOllamaEmbeddings,
    )
    EmbeddingService(EmbeddingType.OLLAMA, "qwen3-0.6b")
    assert captured["keep_alive"] == -1
    assert "num_thread" not in captured
    assert "num_gpu" not in captured
    assert "temperature" not in captured
    assert captured["model"] == "qwen3-embedding:0.6b"
