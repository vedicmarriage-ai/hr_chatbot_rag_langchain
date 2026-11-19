# HR Chatbot — Service Contract (v2 proposal)

**Status:** Proposal (current `/api/ask` is v1: `question` + unused `conversation_id`)  
**Last updated:** 2026-11-19  
**Related:** VDM `src/modules/chatbot/docs/CHATBOT_MODULE_PROPOSAL.md`; design library `docs/CHATBOT_V2_IMPLEMENTATION_PROPOSAL.md`

This service is a **private RAG generator**. It owns **document retrieval** (Chroma + BM25) and **answer generation** (Sarvam gateway / Ollama fallback). It does **not** own end-user conversations, timestamps, thumbs, or Mongo.

Production caller: **VedicMarriage Node backend** only. The browser must not call this process.

---

## 1. Current gaps (v1)

| Topic | Today |
|-------|--------|
| `conversation_id` | Echoed as `metadata.session_id`; **not** used to isolate memory |
| Memory | One process-wide `ConversationBufferMemory` on a singleton `RAGService` — concurrent widgets share turns |
| History in retrieve | Last question only |
| Feedback | None |
| Auth | None; CORS `http://localhost:3000` (dev leftover) |
| Persistence | Chroma for **docs** only |

Streamlit may keep using `RAGService` in-process. HTTP `/api/ask` must become **stateless per request**.

---

## 2. Proposed `POST /api/ask`

### Request

```json
{
  "question": "What about refunds?",
  "conversation_id": "674a…",
  "history": [
    { "role": "user", "content": "Is registration free?" },
    { "role": "assistant", "content": "Registration on Vedic Marriage website is FREE." }
  ],
  "feedback": [
    {
      "value": "down",
      "comment": "Did not mention if premium is paid",
      "message_excerpt": "Registration on Vedic Marriage website is FREE."
    }
  ]
}
```

| Field | Required | Notes |
|-------|----------|--------|
| `question` | yes | Current user turn |
| `conversation_id` | no | Tracing / logs only; **do not** look up server memory |
| `history` | no | Prior turns, oldest first; backend truncates (e.g. last 8). Roles `user` \| `assistant` |
| `feedback` | no | Ratings for prior assistant turns in this window |

Ignore unused `max_tokens` / `temperature` on the product path unless explicitly wired to generation settings.

### Response (unchanged shape)

```json
{
  "answer": "…",
  "sources": ["FAQ.docx …"],
  "metadata": {
    "session_id": "674a…",
    "provider": "sarvam",
    "latency_ms": 1621.54,
    "prompt_tokens": 612,
    "completion_tokens": 768,
    "total_tokens": 1380,
    "generated_at": "2026-11-19T09:41:02.000Z"
  }
}
```

`session_id` = request `conversation_id` or a new UUID if omitted (backend already has the Mongo id; echo is enough). **`generated_at` is tracing only.** Product bubble times come from VDM Mongo.

`POST /chat` may stay for Streamlit/curl; it should use the same per-request history path, not the global buffer.

---

## 3. Generation behavior

1. **Do not** read or write `ConversationBufferMemory` for HTTP asks.
2. Build prompt `chat_history` from request `history` (role labels, cap chars per turn).
3. Append a **feedback** block when `feedback` is non-empty, e.g.  
   `User marked a previous answer unhelpful (down): "{comment}". Prefer addressing that gap.`
4. **Query rewrite (recommended):** if `history` is non-empty, rewrite `question` with the last user/assistant pair before retrieve (so “what about refunds?” hits refund chunks). Keep rewrite local/cheap; do not add a second Sarvam round unless measured.
5. Retrieve on the rewritten (or raw) question; generate with existing `generate_from_docs`.
6. Empty retrieval → existing grounding refusal; `provider: none`; still return 200.

Replace unused `ChatHistory` model in `app/models/chat.py` with the ask payload types above. Do **not** add `GET` history or `POST` feedback routes here.

---

## 4. Auth and network

When `HR_SERVICE_TOKEN` is set:

- Require `Authorization: Bearer <token>` on `/api/ask` and `/chat`.
- `/health` stays unauthenticated for probes.

CORS: restrict to unused or omit for product (Node is same-host). Do not add the Vite origin as a product client.

Bind `127.0.0.1` in production. Nginx must **not** publish `:8010`.

```env
HR_SERVICE_TOKEN=<same as VDM HR_CHATBOT_SERVICE_TOKEN>
AI_CHAT_BASE_URL=http://127.0.0.1:8091
AI_SERVICE_TOKEN=<existing>
```

---

## 5. Explicit non-goals

- MongoDB (or any conversation store) in this repo
- Thumbs/comment REST
- Per-user session maps in RAM as a substitute for the BFF
- Browser CORS for the design-library origin

---

## 6. Tests

- Two sequential asks with **different** `history` arrays must not leak turns (proves global memory is unused).
- `feedback` with `down` + comment appears in the prompt string (unit, mock generation).
- Missing/invalid bearer → 401 when token configured.
- Fixture docs only; **no live Sarvam**.
