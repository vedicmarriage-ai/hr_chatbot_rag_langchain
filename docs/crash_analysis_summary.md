# Crash / Hang Analysis Summary

**Scope:** Ad-hoc diagnostics in `simple_diagnostic.py` and `crash_diagnostic.py` (untracked helpers).  
**Context:** MacBook (≈8 GB RAM) hung or became unusable during local Ollama inference with larger models.

This was **not** a deep post-mortem of a kernel panic or app exception dump. It was a **simple resource / config analysis** of why the machine froze under LLM load.

---

## What was analyzed

| Check | Purpose |
|-------|---------|
| System RAM / disk (`vm_stat`, `psutil`) | See if memory was exhausted |
| Ollama `/api/tags` | Confirm service + loaded models |
| Python process list | Detect multiple heavy interpreters |
| GPU (CUDA) | N/A on this Mac path — CPU / Apple Silicon via Ollama |
| Chroma size | Rule out huge vector store as the cause |
| Rough model size table | Compare 7B quantized models vs tiny models |
| Optional memory stress alloc | See how quickly RAM hits >90% |

---

## Root cause (conclusion)

**Memory pressure from stacking too much on 8 GB RAM**, especially when using **~7B** Ollama models:

| Component | Approx. RAM |
|-----------|-------------|
| `mistral:7b-…-q2_K` / `llama2:7b-…` | ~2.7–4 GB weights |
| KV cache (large `num_ctx`) | ~1–2 GB |
| Embeddings (e.g. HF `bge-base` + PyTorch) | ~1–1.5 GB |
| Streamlit / Python / macOS | ~2–3 GB |

Total often **exceeds 8 GB** → swap thrash → UI/system **hang** (felt like a “crash”).

Vector store size was small (hundreds of MB) and was **not** the primary cause.

---

## Recommendations from that analysis

1. Prefer **small** local LLMs (~1 GB class), not 7B, on 8 GB machines.  
2. Use **quantized** tags when a larger model is required.  
3. Cap context / generation (`num_ctx`, `num_predict`); avoid huge prompts.  
4. Close extra Python / Streamlit / Ollama loads before heavy inference.  
5. Restart Ollama if a model load left the machine in a bad state.

---

## What we did afterward (project state)

- Default generation moved to **`qwen2.5:1.5b`** (~1 GB) with tighter LLM params.  
- Embeddings prefer lighter Ollama options (e.g. **`qwen3-embedding:0.6b`**) where possible.  
- Stacks are curated in `app/core/rag_stacks.py` for 8 GB–friendly combos.

Those diagnostic scripts remain optional local tools; they are **not** required for production runtime.

---

## Note on later Streamlit `torch.classes` tracebacks

Separate from this hang analysis: Streamlit file-watcher noise with PyTorch (`Tried to instantiate class '__path__._path'`) is usually **harmless log spam**, not the same OOM hang issue.
