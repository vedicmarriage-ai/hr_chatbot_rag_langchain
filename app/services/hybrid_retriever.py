"""Hybrid dense (embeddings) + sparse (BM25) candidate retrieval."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import get_settings

settings = get_settings()

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _doc_key(doc: Document) -> str:
    faq_id = doc.metadata.get("faq_id")
    if faq_id:
        return f"faq:{faq_id}"
    legal_id = doc.metadata.get("legal_id")
    if legal_id:
        return f"legal:{legal_id}"
    source = doc.metadata.get("source", "")
    section = doc.metadata.get("section", "")
    return f"{source}|{section}|{hash(doc.page_content[:200])}"


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Document]:
    """Merge ranked lists with (optionally weighted) RRF; preserve first-seen Document."""
    scores: dict[str, float] = {}
    best_doc: dict[str, Document] = {}
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")

    for list_weight, ranked in zip(weights, ranked_lists):
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + float(list_weight) * (
                1.0 / (k + rank)
            )
            if key not in best_doc:
                best_doc[key] = doc

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [best_doc[key] for key, _ in ordered]


class BM25Index:
    """In-memory BM25 over corpus chunks loaded from the vector store."""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self._corpus_tokens = [_tokenize(d.page_content) for d in documents]
        self._bm25: Optional[BM25Okapi] = None
        if documents:
            self._bm25 = BM25Okapi(self._corpus_tokens)

    @classmethod
    def from_vectorstore(cls, vectorstore: Any) -> "BM25Index":
        raw = vectorstore.get(include=["documents", "metadatas"])
        docs: List[Document] = []
        contents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        for text, meta in zip(contents, metadatas):
            if not text:
                continue
            docs.append(Document(page_content=text, metadata=meta or {}))
        return cls(docs)

    def search(self, query: str, k: int) -> List[Document]:
        if not self._bm25 or not self.documents:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self.documents[i] for i in ranked_idx if scores[i] > 0]


def hybrid_search(
    vectorstore: Any,
    bm25_index: Optional[BM25Index],
    query: str,
    fetch_k: int,
    use_hybrid: bool = True,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
) -> List[Document]:
    """Dense search, optionally fused with BM25 via weighted RRF."""
    dense_w = (
        settings.HYBRID_DENSE_WEIGHT if dense_weight is None else float(dense_weight)
    )
    sparse_w = (
        settings.HYBRID_SPARSE_WEIGHT if sparse_weight is None else float(sparse_weight)
    )
    # Normalize to keep an equal 50/50 semantics even if both are set to 0.5
    total = dense_w + sparse_w
    if total <= 0:
        dense_w, sparse_w = 0.5, 0.5
    else:
        dense_w, sparse_w = dense_w / total, sparse_w / total

    dense_k = fetch_k
    if use_hybrid and bm25_index is not None:
        dense_k = max(fetch_k, settings.HYBRID_DENSE_K)

    dense_docs = vectorstore.similarity_search(query, k=dense_k)

    if not use_hybrid or bm25_index is None:
        return dense_docs[:fetch_k]

    sparse_docs = bm25_index.search(query, k=max(fetch_k, settings.HYBRID_SPARSE_K))
    fused = reciprocal_rank_fusion(
        [dense_docs, sparse_docs],
        weights=[dense_w, sparse_w],
    )
    return fused[:fetch_k]
