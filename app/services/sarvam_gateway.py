"""Call vedicmarriage_ai chat. This process does not hold a Sarvam key."""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class SarvamGatewayError(Exception):
    """AI service unreachable, timed out, or returned 503."""


@dataclass
class GatewayAnswer:
    content: str
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    provider: str = "sarvam"


def complete(prompt: str) -> GatewayAnswer:
    settings = get_settings()
    base = (settings.AI_CHAT_BASE_URL or "").rstrip("/")
    if not base:
        raise SarvamGatewayError("AI_CHAT_BASE_URL is not set")

    payload = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.AI_SERVICE_TOKEN}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.AI_CHAT_TIMEOUT_SECONDS) as response:
            if response.status >= 500:
                raise SarvamGatewayError(f"AI chat status {response.status}")
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code >= 500:
            raise SarvamGatewayError(f"AI chat status {exc.code}") from exc
        raise
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise SarvamGatewayError(str(exc)) from exc

    usage = body.get("usage") or {}
    return GatewayAnswer(
        content=str(body.get("content") or ""),
        latency_ms=body.get("latency_ms"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
