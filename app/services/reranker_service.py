"""Cross-encoder reranking for retrieval — replaces query-specific heuristics."""

import re
from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.core.config import get_settings

settings = get_settings()

_QA_PATTERN = re.compile(r"Q\d+\.", re.IGNORECASE)
_REGISTRATION_QUERY = re.compile(
    r"\b(register|registration|sign\s*up|signup|join|create\s+(?:an?\s+)?account|new\s+account)\b",
    re.IGNORECASE,
)
_REGISTRATION_QUESTION = re.compile(
    r"\b(register|sign[- ]?up|sign-up|submit)\b",
    re.IGNORECASE,
)
_MEMBER_CONTACT_QUESTION = re.compile(
    r"\b(get in touch with other members|contacting members|contact members|reach out to members)\b",
    re.IGNORECASE,
)
_LOGIN_QUESTION = re.compile(r"\b(login|logout|password|forgot)\b", re.IGNORECASE)


@lru_cache
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.RERANK_MODEL)


def _is_qa_chunk(doc: Document) -> bool:
    """Detect FAQ-style Q&A chunks vs long policy/legal prose."""
    text = doc.page_content
    head = text[:500]
    return bool(_QA_PATTERN.search(head)) and "Ans." in head


def _faq_intent_adjustment(query: str, doc: Document) -> float:
    """Nudge scores so registration queries prefer registration FAQ, not member contact."""
    faq_question = str(doc.metadata.get("faq_question", ""))
    if not faq_question:
        q_match = re.search(r"Q\d+\.\s*(.+?)(?:\n|Ans)", doc.page_content, re.IGNORECASE)
        faq_question = q_match.group(1) if q_match else ""

    if not faq_question:
        return 0.0

    question = faq_question.lower()
    q = query.lower()
    adj = _overlap_ratio(query, faq_question) * 0.2

    if not _REGISTRATION_QUERY.search(q):
        return adj

    if _REGISTRATION_QUESTION.search(question):
        adj += 0.35
    if _MEMBER_CONTACT_QUESTION.search(question):
        adj -= 0.45
    if _LOGIN_QUESTION.search(question) and not re.search(
        r"\b(login|logout|password|forgot)\b", q
    ):
        adj -= 0.15

    source = doc.metadata.get("source", "").lower()
    if _REGISTRATION_QUERY.search(q):
        if doc.metadata.get("chunk_type") == "faq":
            adj += 0.15
        elif any(token in source for token in ("terms", "refund", "policy")):
            adj -= 0.35

    return adj


def _prioritize_for_query(query: str, docs: List[Document]) -> List[Document]:
    """Put the most intent-aligned FAQ chunk first for small-LLM context windows."""
    if not _REGISTRATION_QUERY.search(query):
        return sorted(docs, key=lambda doc: len(doc.page_content))

    def sort_key(doc: Document) -> tuple:
        faq_question = str(doc.metadata.get("faq_question", "")).lower()
        is_registration_faq = (
            doc.metadata.get("chunk_type") == "faq"
            and bool(_REGISTRATION_QUESTION.search(faq_question))
        )
        is_faq = doc.metadata.get("chunk_type") == "faq"
        is_member_contact = bool(_MEMBER_CONTACT_QUESTION.search(faq_question))
        return (
            0 if is_registration_faq else 1,
            0 if is_faq else 1,
            1 if is_member_contact else 0,
            len(doc.page_content),
        )

    return sorted(docs, key=sort_key)


def _apply_qa_bonus(
    ranked: List[Tuple[float, Document]],
) -> List[Tuple[float, Document]]:
    """Slight score boost for structured Q&A chunks — not a hard filter."""
    bonus = settings.RERANK_QA_BONUS
    return sorted(
        (
            (score + bonus if _is_qa_chunk(doc) else score, doc)
            for score, doc in ranked
        ),
        key=lambda item: item[0],
        reverse=True,
    )


def _overlap_ratio(left: str, right: str) -> float:
    left_words = set(left.lower().split())
    right_words = set(right.lower().split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / min(len(left_words), len(right_words))


def _select_diverse_top_k(
    ranked: List[Tuple[float, Document]],
    k: int,
    max_overlap: float = 0.85,
    max_per_source: int = 2,
) -> List[Document]:
    """Keep high-scoring chunks; skip near-duplicates and over-represented sources."""
    selected: List[Document] = []
    selected_texts: List[str] = []
    source_counts: dict[str, int] = {}
    long_source_counts: dict[str, int] = {}
    long_threshold = settings.RERANK_LONG_CHUNK_CHARS

    for _, doc in ranked:
        text = doc.page_content.strip()
        source = doc.metadata.get("source", "unknown")

        if any(_overlap_ratio(text, seen) > max_overlap for seen in selected_texts):
            continue
        if source_counts.get(source, 0) >= max_per_source:
            continue
        if (
            len(text) > long_threshold
            and long_source_counts.get(source, 0) >= 1
        ):
            continue

        selected.append(doc)
        selected_texts.append(text)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(text) > long_threshold:
            long_source_counts[source] = long_source_counts.get(source, 0) + 1
        if len(selected) >= k:
            break

    return selected


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int,
) -> List[Document]:
    if not documents:
        return []

    if len(documents) <= top_n:
        return documents

    model = get_cross_encoder()
    raw_scores = model.predict([(query, doc.page_content) for doc in documents])
    max_chars = settings.RERANK_MAX_CHUNK_CHARS
    ranked = sorted(
        (
            (
                float(score)
                - min(len(doc.page_content) / max_chars, 0.5)
                + _faq_intent_adjustment(query, doc),
                doc,
            )
            for score, doc in zip(raw_scores, documents)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    ranked = _apply_qa_bonus(ranked)
    selected = _select_diverse_top_k(
        ranked,
        top_n,
        max_per_source=settings.RERANK_MAX_CHUNKS_PER_SOURCE,
    )
    return _prioritize_for_query(query, selected)
