# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version is defined in `app/core/version.py` and exposed via the API root and `/health` endpoints.

## [1.2.4] - 2026-11-19

### Changed

- Chat history for generation is the last **3 turns** per `conversation_id`
  (280 characters each, source footers stripped). The process-wide
  `ConversationBufferMemory` is no longer mixed into `/api/ask` prompts.
- If Sarvam returns empty visible text (policy 502), retry once with a compact
  prompt: no history, three-sentence cap.
- Canned welcome / `hi` turns are not stored in RAG chat history, so the next
  real question is not prompted with the opener.

## [1.2.3] - 2026-11-19

### Changed

- Ollama embeddings no longer set `num_thread=8` / `num_gpu=-1` (those made
  warm query embeds ~9s on a 4-vCPU host). `keep_alive=-1` keeps 0.6B loaded.
- Generation prompt uses the top 3 chunks, 400 characters each.
- Local `qwen2.5:1.5b` fallback is **off** by default (`LOCAL_LLM_FALLBACK=false`)
  so a Sarvam miss does not evict the embedding model. `/api/ask` returns **502**.
- Retrieve candidates: FAQ 12, policy 16, hybrid k 12, `TOP_K` 3.
- `POST /api/ask` and `/chat` run retrieval+generation in a threadpool.
- Metadata includes `retrieve_ms` and `generate_ms`. Startup warms MiniLM + one embed.
- The AI chat silo accepts optional `max_tokens`; HR does not send it yet because
  256–1024 on the full RAG prompt returned empty Sarvam content.

### Docs

- [`docs/LATENCY_PROPOSAL.md`](docs/LATENCY_PROPOSAL.md) phase 1 implemented.

## [1.2.2] - 2026-11-19

### Added

- Greeting short-circuit (`hi` / `hello` / `namaste` / …) returns a VedicMarriage Bot
  welcome without retrieval or Sarvam (`app/services/greetings.py`).
- RAG prompt identifies the speaker as **VedicMarriage Bot**.

## [1.2.1] - 2026-11-19

### Added

- Product nginx snippet `deploy/nginx-hr-chat.inc`: same-origin `POST /hr-chat/api/ask` → loopback `:8010`. Without this, SPA nginx returns **405** and the widget shows unavailable.
- Optional systemd unit `deploy/hr-chatbot.service`.
- `CORS_ALLOW_ORIGINS` (comma-separated) with production/staging defaults. Same-origin proxy does not rely on CORS.
- [`docs/PRODUCTION.md`](docs/PRODUCTION.md).

## [1.2.0] - 2026-11-19

### Added

- Sarvam generation through the VedicMarriage AI service (`app/services/sarvam_gateway.py`). Retrieval stays local. The Sarvam key is not stored in this app.
- Fallback to `qwen2.5:1.5b` on timeout, connection error, missing `AI_CHAT_BASE_URL`, or HTTP 503 (`LOCAL_LLM_FALLBACK`).
- Empty retrieval still returns the refusal and calls neither model.
- `/api/ask` metadata: `provider`, `latency_ms`, and token counts. Ollama fallback leaves tokens empty.
- Docs: [`docs/SARVAM_GENERATION.md`](docs/SARVAM_GENERATION.md). Tests: `tests/test_answer_generation.py`.
- Hybrid BM25 + embedding retrieval with reciprocal rank fusion.
- Policy-query aware fetch depth and legal rerank bonuses (no FAQ Q&A bonus on policy queries).
- Stronger legal chunk headers/`legal_id` metadata and `strengthen_legal_qrels.py`.
- Hybrid weight tuning defaults: dense 0.3 / BM25 0.7 (always-on); eval script `--sweep`.
- Source citations with chunk quotes (`faq_id` / `legal_id` + excerpt), not filenames only.
- Generic numeric grounding check and a harder refusal prompt. LLM temperature default is 0.
- Retrieval-improvement interview notebook (`notebook/retrieval_improvement.ipynb`).

### Changed

- `RAGService.process_message` retrieves first, then `generate_from_docs`, instead of `ConversationalRetrievalChain` for the answer.
- Chain and memory imports use `langchain-classic` (`requirements.txt`).

## [1.1.0] - 2026-07-09

### Added

- Cross-encoder reranking (`app/services/reranker_service.py`) with FAQ intent alignment for registration-style queries.
- Structure-aware document chunking: FAQ sections with unique `faq_id` metadata; legal docs by section/clause.
- RAG stack configuration (`app/core/rag_stacks.py`) for validated LLM + embedding combinations.
- Retrieval benchmark pipeline with IR metrics (Precision@k, Recall@k, MRR, MAP, nDCG@k).
- Qrels workflow: OpenAI generators (FAQ, all docs, paraphrases, composite), validator, and FAQ `gold_faq_id` patcher.
- Composite multi-chunk evaluation via `data/benchmark_results/composite_queries.json` (config-driven, not runtime hardcoding).
- `scripts/run_hr_bot.sh run-api` for starting FastAPI with the conda environment.
- Centralized version module (`app/core/version.py`) and this changelog.

### Changed

- `DocumentService` uses line-by-line FAQ parsing so section headers (Registration, Login, Payment, etc.) are distinct.
- `RAGService` uses `RerankRetriever` instead of query-specific retrieval hacks; simplified HR prompt.
- `benchmark_vector_stores.py` supports `gold_faq_id` matching and relaxed keyword checks when chunk ID is set.
- Vector store rebuild wipes persist directory before indexing to avoid stale Chroma chunks.
- `requirements.txt`: added `openai`, `python-dotenv`; benchmark and qrels scripts load `.env.dev` automatically.

### Fixed

- FAQ chunks no longer share one section label (`Getting started:Registration`) across all `Q1` items.
- “How do I join / register?” queries retrieve `registration-Q1` instead of member-contact FAQ noise.
- Benchmark reads `metadata.json` for Ollama store embedding config (fixes dimension / model key mismatches).
- Word lock files (`~$*.docx`) excluded from document loading.

## [1.0.0] - 2025-05-08

### Added

- Initial HR chatbot: FastAPI + Streamlit, LangChain RAG, ChromaDB, Ollama LLM and embeddings.
- Document ingestion from `data/docs/Text_documents/`.
- Basic vector store creation and chat API.

[Unreleased]: https://github.com/vedicmarriage-ai/hr_chatbot_rag_langchain/compare/v1.1.0...HEAD
[1.2.0]: https://github.com/vedicmarriage-ai/hr_chatbot_rag_langchain/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/vedicmarriage-ai/hr_chatbot_rag_langchain/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/vedicmarriage-ai/hr_chatbot_rag_langchain/releases/tag/v1.0.0
