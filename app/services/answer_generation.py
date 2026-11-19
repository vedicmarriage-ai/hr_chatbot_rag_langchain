"""Generate an HR answer from already retrieved chunks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.sarvam_gateway import GatewayAnswer, SarvamGatewayError, complete

logger = logging.getLogger(__name__)


@dataclass
class GenerationRecord:
    provider: str
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    text: str


def _context(docs: list) -> str:
    settings = get_settings()
    max_docs = max(1, settings.GENERATE_MAX_CHUNKS)
    max_chars = max(1, settings.GENERATE_MAX_CHUNK_CHARS)
    parts: list[str] = []
    for doc in docs[:max_docs]:
        text = (getattr(doc, "page_content", None) or "").strip()
        if text:
            parts.append(text[:max_chars])
    return "\n\n".join(parts)


def generate_from_docs(
    *,
    question: str,
    docs: list,
    prompt_template,
    chat_history: str,
    refusal: str,
    local_generate,
    fallback_enabled: bool,
) -> GenerationRecord:
    """Return a refusal without calling a model when retrieval found nothing."""
    if not docs or not _context(docs).strip():
        logger.info(
            "hr_generation provider=none latency_ms=0 prompt_tokens=None completion_tokens=None total_tokens=None"
        )
        return GenerationRecord("none", 0, None, None, None, refusal)

    context = _context(docs)
    prompt = prompt_template.format(
        context=context,
        chat_history=chat_history or "",
        question=question,
    )
    compact = (
        "You are VedicMarriage Bot. Answer in at most three short sentences. "
        "Use only the Context. No scratchpad.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        remote: GatewayAnswer = complete(prompt)
        if not (remote.content or "").strip():
            raise SarvamGatewayError("empty visible content")
    except SarvamGatewayError as exc:
        logger.info("hr_generation_compact_retry reason=%s", exc)
        try:
            remote = complete(compact)
            if not (remote.content or "").strip():
                raise SarvamGatewayError("empty visible content after compact retry")
        except SarvamGatewayError as retry_exc:
            if not fallback_enabled:
                raise retry_exc
            logger.info("hr_generation_fallback reason=%s", retry_exc)
            started = time.perf_counter()
            text = local_generate(compact)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "hr_generation provider=ollama latency_ms=%s prompt_tokens=None completion_tokens=None total_tokens=None",
                latency_ms,
            )
            return GenerationRecord("ollama", latency_ms, None, None, None, text)

    logger.info(
        "hr_generation provider=sarvam latency_ms=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        remote.latency_ms,
        remote.prompt_tokens,
        remote.completion_tokens,
        remote.total_tokens,
    )
    return GenerationRecord(
        "sarvam",
        remote.latency_ms,
        remote.prompt_tokens,
        remote.completion_tokens,
        remote.total_tokens,
        remote.content,
    )
