"""Generate an HR answer from already retrieved chunks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

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
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)


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

    prompt = prompt_template.format(
        context=_context(docs),
        chat_history=chat_history or "",
        question=question,
    )
    try:
        remote: GatewayAnswer = complete(prompt)
    except SarvamGatewayError as exc:
        if not fallback_enabled:
            raise
        logger.info("hr_generation_fallback reason=%s", exc)
        started = time.perf_counter()
        text = local_generate(prompt)
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
