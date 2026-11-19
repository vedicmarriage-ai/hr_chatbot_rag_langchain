# HR chatbot — latency proposal

**Status:** Phase 1 implemented in **1.2.3** (embed options, keep_alive, slim prompt, no local fallback by default, threadpool, retrieve/generate timings, warmup). Live FAQ after deploy: **~2.3 s** wall (`retrieve_ms` ~700, `generate_ms` ~1.6). Policy asks can still **502** when Sarvam spends the 2048-token budget on hidden reasoning and returns empty `content`. Phase 2: durable silo image with a working `max_tokens` on long RAG prompts.  
**Date:** 2026-11-19  
**Host:** 4 vCPU KVM (AMD EPYC 9354P slice), Ollama 0.40.16 CPU-only, Qwen3-embedding 0.6B Q8 resident ~1.72 GB, MiniLM rerank CPU, Sarvam 105B via `:8091`.

`metadata.latency_ms` is **generation only**. The widget waits on the full `POST /api/ask`.

---

## Target

| Path | Today (this host) | Target (p50, warm) |
|------|-------------------|---------------------|
| Greeting (`hi`) | canned, ~0s | keep |
| FAQ (registration) | **~12–15s** retrieve + ~2s Sarvam (or 30s+ if fallback) | **3–4s** |
| Policy (refund) | same retrieve + ~2.5–4s Sarvam | **3–4s** warm |
| Sarvam down | local 1.5B ~35s + embed eviction | **fail in &lt;1s** (existing toast) |

Do not rebuild the 105-chunk Qwen3 0.6B index for speed.

---

## Measured split (this session)

Isolated timings on the live Ollama + Chroma store. Model was **already loaded** (`ollama ps`: `qwen3-embedding:0.6b`, `size_vram=0`).

### Retrieve (current LangChain options)

`EmbeddingService` always sends `num_ctx=8192`, `num_gpu=-1`, `num_thread=8`, `temperature=0`.

| Step | Cold (this run) | Repeat |
|------|-----------------|--------|
| `embed_query` | 30.2 s | **8.6–13.8 s** |
| Chroma `similarity_search(k=20)` (embeds again) | — | **~9.8–10.0 s** |
| BM25 only | — | **0.1–0.3 ms** |
| Hybrid (dense + BM25) | — | **~9.0–9.8 s** (dominated by the dense embed) |
| MiniLM rerank 20 → 5 | 1.3 s | **0.67 s** |
| MiniLM rerank 12 → 5 | — | **0.37–0.44 s** |
| MiniLM rerank 8 → 5 | — | **0.20–0.21 s** |

Hybrid does **not** add meaningful CPU; BM25 is free. The 9 s “retrieve” is one Ollama embed inside Chroma.

### Raw `/api/embed` (same model, same host)

| Options | 1st after option change | Steady |
|---------|-------------------------|--------|
| **no options / keep_alive=-1** | 1.7 s (load) then 47 ms | **45–58 ms** |
| `num_ctx` 256 / 512 / 2048 / 8192, default threads | ~1.1–1.9 s (reload) | **38–51 ms** |
| `num_thread=8` + `num_ctx=512` | 29 s | **~9.8–9.9 s** |
| **Current LangChain set** (`num_ctx=8192`, `num_gpu=-1`, `num_thread=8`, `temperature=0`) | 31 s | **8.8–9.2 s** |

Root cause: **`num_thread=8` on a 4-vCPU VM**. Oversubscription, not model size, not `num_ctx=8192`, not keep-alive. Leave thread count to Ollama (physical cores). Do not set `num_gpu=-1` on this CPU box.

### Generate (Sarvam, same prompts as RAG)

Registration FAQ, 5×800-char vs 3×400-char context:

| Prompt | Wall | Completion tokens | Answer quality (spot) |
|--------|------|-------------------|------------------------|
| Full (~1942 chars, 5 chunks) | **2.15 s** | 998 | “Registration … is FREE.” |
| Slim (~765 chars, 3×400) | **0.94 s** | 317 | Same fact |

Refund:

| Prompt | Wall | Completion tokens |
|--------|------|-------------------|
| Full (~2106 chars) | **2.71 s** | 993 |
| Slim (~1155 chars) | **1.80 s** | 714 |

`:8091` hardcodes `max_tokens=2048` and does **not** accept a cap from HR. Slimming the prompt already cuts FAQ generation to ~1 s. Policy still ~1.8 s until the silo caps completions (~256 FAQ / ~512 policy).

---

## Why 3–4 s is reachable (budget)

Warm path after dropping `num_thread=8` and keeping 0.6B loaded:

| Step | Budget | How |
|------|--------|-----|
| Query embed | **50 ms** | Ollama default threads + `keep_alive=-1` |
| Dense search (second embed inside Chroma) | **50 ms** | Same; optional: `similarity_search_by_vector` to skip the duplicate |
| BM25 + RRF | **&lt;1 ms** | Already |
| MiniLM rerank 8 candidates | **200 ms** | `fetch_k` 20→8 (12 if eval needs it) |
| Sarvam (slim prompt) | **900–1800 ms** | 3×400 chars; later `max_tokens` in silo |
| HTTP / FastAPI | **&lt;50 ms** | |
| **Total** | **~1.3–2.2 s FAQ / ~2.1–2.8 s policy** | Headroom to **3–4 s p50** |

End-to-end today is **~12 s+** because retrieve alone is ~9 s + rerank ~0.7 s + Sarvam ~1–3 s. Fixing embed options removes ~9 s. That is the 3–4 s path. A bigger embedding model would make this worse on 4 vCPU.

Sub-second (~0.75 s) is **not** this stack: Sarvam 105B is already 0.9–1.8 s after slim. That needs extractive FAQ / TTL cache / streaming first token — phase 3.

---

## Why it is slow (ordered, updated)

1. **`num_thread=8` on 4 vCPU** in `OllamaEmbeddings` — ~9 s per embed even when the 1.72 GB model is resident. Largest cost on a healthy daemon.
2. **Ollama fallback on the same daemon.** `LOCAL_LLM_FALLBACK=true` loads `qwen2.5:1.5b` after a 30 s Sarvam timeout and **evicts** 0.6B. Next question pays 30 s reload **and** 9 s embeds again. Widget already has “unavailable”.
3. **Oversized generation.** Five 800-char chunks + silo `max_tokens=2048` → ~1000 completion tokens. FAQ was 2.1 s vs 0.9 s slim.
4. **MiniLM over 20 docs ~0.7 s.** Corpus is 105 chunks; 8–12 candidates is enough.
5. **Sync FastAPI** blocking `process_message` — queueing, not first-byte of a single ask.
6. **`keep_alive` unset** — default 5 m; fine if traffic exists, still set `-1` so idle nights stay warm (~1.7 s reload vs 50 ms).
7. Dead weight: `LLMService` + `ConversationalRetrievalChain` at startup (not on the answer path).

Embeddings are **not** too large. Do not upgrade to Qwen3-Embedding 4B/8B.

---

## Phase 1 — product (same day, low risk)

| Change | Where | Effect |
|--------|--------|--------|
| **Stop passing `num_thread` / `num_gpu` / `temperature` on embeddings** | `embedding_service.py` | 9 s → **~50 ms** embed. Optional `num_ctx=512` for queries (index already built). |
| **`keep_alive=-1`** | `OllamaEmbeddings` | Keep 0.6B resident (~1.72 GB RAM). |
| **`LOCAL_LLM_FALLBACK=false` in production** | `.env` | No 35 s local generate, no eviction. |
| **Pre-warm at startup** | `RAGService.__init__` | One `embed_query("warmup")` + `get_cross_encoder()`. |
| **Cap context: top 3 chunks, 400 chars each** | `generate_from_docs` | FAQ Sarvam ~1 s in this measurement. |
| **Log `retrieve_ms` and `generate_ms`** | `process_message` | Prove the split in `/api/ask` metadata. |
| **`run_in_threadpool` for `process_message`** | `routes.py` | Health and greetings not blocked. |

Expect FAQ p50 **~2 s** and policy **~3 s** warm on this host after phase 1, with 3–4 s as a comfortable SLO.

## Phase 2 — retrieve trim + silo cap

| Change | Default to try | Guardrail |
|--------|----------------|-----------|
| FAQ `fetch_k` | 20 → **8–12** | Registration still ranks gold FAQ. |
| Policy `fetch_k` | 35 → **12–16** | Refund/terms qrels. |
| `TOP_K_RESULTS` | 5 → **3** | Prompt size. |
| Hybrid `DENSE_K`/`SPARSE_K` | 25 → **12** | |
| `max_tokens` on `:8091` | 256 FAQ / 512 policy | Today hardcoded 2048; HR body ignored. |
| `similarity_search_by_vector` | skip second embed | Extra ~50 ms only. |

Do not drop hybrid without a qrels pass (`data/benchmark_results/`).

## Phase 3 — architecture (with BFF)

- One Ollama **only** for `qwen3-embedding:0.6b`. No chat model on that daemon.
- Sarvam timeout **12–15 s**; on failure return 502 JSON (frontend toast). No local LLM.
- Optional **exact-question cache** (TTL 10–15 min) if product wants sub-second repeats.
- Optional **SSE/stream** later so the bubble starts before the last token.

---

## Explicit non-goals

- Re-embedding the 105 chunks in a larger model for speed.
- GPU for MiniLM (nice later; after embed is fixed, rerank is ~200 ms).
- Mongo / conversation store (unrelated to first-token time).

---

## Suggested first patch

1. Drop `num_thread=8` / `num_gpu=-1` / `temperature=0` from `OllamaEmbeddings`; set `keep_alive=-1`.
2. Production `.env`: `LOCAL_LLM_FALLBACK=false`.
3. Prompt: 3 chunks × 400 chars; log retrieve vs generate.

That is how this host hits **3–4 seconds**. The 0.6B model is already fast enough; the client is tying it in knots.
