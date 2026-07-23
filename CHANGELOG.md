# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version is defined in `app/core/version.py` and exposed via the API root and `/health` endpoints.

## [Unreleased]

### Added
- Hybrid BM25 + embedding retrieval with reciprocal rank fusion
- Policy-query aware fetch depth and legal rerank bonuses (no FAQ Q&A bonus on policy queries)
- Stronger legal chunk headers/`legal_id` metadata and `strengthen_legal_qrels.py`
- Hybrid weight tuning defaults: dense 0.3 / BM25 0.7 (always-on); eval script `--sweep`
- Source citations with chunk quotes (`faq_id` / `legal_id` + excerpt), not filenames only
- Generic numeric grounding check + harder refusal prompt; LLM temperature default 0
- Retrieval-improvement interview notebook (`notebook/retrieval_improvement.ipynb`)

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
[1.1.0]: https://github.com/vedicmarriage-ai/hr_chatbot_rag_langchain/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/vedicmarriage-ai/hr_chatbot_rag_langchain/releases/tag/v1.0.0
