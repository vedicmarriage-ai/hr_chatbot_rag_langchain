from app.core.config import cors_allow_origins


def test_cors_allow_origins_splits_and_strips():
    origins = cors_allow_origins(" https://staging.vedicmarriage.com , http://localhost:9101 ")
    assert origins == [
        "https://staging.vedicmarriage.com",
        "http://localhost:9101",
    ]


def test_cors_allow_origins_empty():
    assert cors_allow_origins("") == []
