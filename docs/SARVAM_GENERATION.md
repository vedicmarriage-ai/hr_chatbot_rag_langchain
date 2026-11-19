# Sarvam generation

Retrieval stays in this repo. The answer step calls the VedicMarriage AI service, which holds the Sarvam key.

## When each model runs

1. Retrieve locally (Chroma, hybrid BM25 + dense, cross-encoder rerank).
2. If no chunk text comes back, return `GROUNDING_REFUSAL_MESSAGE` and call no model.
3. Otherwise `POST {AI_CHAT_BASE_URL}/v1/chat/completions` with the grounded prompt.
4. On a missing URL, timeout, connection error, or HTTP 503, return **502** unless `LOCAL_LLM_FALLBACK=true` (then local `qwen2.5:1.5b`). Default is **false** so fallback cannot evict the embedding model.

A weak retrieval result is not a Sarvam outage. If chunks come back, Sarvam still answers and can refuse from the prompt.

## Environment

```env
AI_CHAT_BASE_URL=http://127.0.0.1:8091
AI_SERVICE_TOKEN=<same token as the AI service>
AI_CHAT_TIMEOUT_SECONDS=30
LOCAL_LLM_FALLBACK=false
LOCAL_FALLBACK_MODEL=qwen2.5:1.5b
OLLAMA_EMBED_KEEP_ALIVE=-1
```

`.env` is gitignored. Do not put `SARVAM_API_KEY` in this app.

## API

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | status and version |
| POST | `/api/ask` | `question`; metadata has `provider`, `latency_ms`, `retrieve_ms`, `generate_ms`, and token counts |
| POST | `/chat` | `message` and optional `session_id` |

`provider` is `sarvam`, `ollama`, or `none`. Ollama fallback leaves token counts empty. Sarvam latency and tokens are copied from the AI service response.

## Local process

Default API port in `main.py` is 8000. The checked-out process on this host is `127.0.0.1:8010`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5:1.5b
set -a && source .env && set +a
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8010
```

Embedding `qwen3-embedding:0.6b` is about 639 MB on disk and about 1.7 GB resident while a query is embedded. The 1.5B model stays on disk until a fallback call.

## Tests

`tests/test_answer_generation.py` mocks the gateway and the local model. It does not call Sarvam or Ollama.
