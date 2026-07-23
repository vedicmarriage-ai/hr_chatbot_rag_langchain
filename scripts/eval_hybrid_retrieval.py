"""
Compare dense-only vs hybrid (50/50 weighted RRF) retrieval MRR.

Focus report: FAQ vs Terms vs Refund for one rebuilt store.

Example:
  python scripts/eval_hybrid_retrieval.py \\
    --dataset-path data/benchmark_results/hr_qrels_full_first_pass_for_review.csv \\
    --persist-dir data/chroma_db/ollama_qwen3-0.6b \\
    --dense-weight 0.5 --sparse-weight 0.5 --top-k 5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain_chroma import Chroma

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.core.config import get_settings
from app.services.embedding_service import EmbeddingService, EmbeddingType
from app.services.hybrid_retriever import BM25Index, hybrid_search

settings = get_settings()


def load_qrels(path: Path):
    df = pd.read_csv(path).fillna("")
    qrels_by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    query_texts: Dict[str, str] = {}
    query_source: Dict[str, str] = {}

    for row in df.to_dict(orient="records"):
        query_id = str(row["query_id"]).strip()
        query = str(row["query"]).strip()
        gold_source = str(row["gold_source"]).strip()
        if not query_id or not query or not gold_source:
            continue
        try:
            grade = int(row["relevance_grade"])
        except (TypeError, ValueError):
            grade = 1
        keywords = [
            k.strip().lower()
            for k in str(row["gold_keywords"]).split("|")
            if k.strip()
        ]
        qrels_by_query[query_id].append(
            {
                "gold_source": gold_source.lower(),
                "gold_keywords": keywords,
                "gold_section_regex": str(row.get("gold_section_regex", "")).strip(),
                "gold_faq_id": str(row.get("gold_faq_id", "")).strip(),
                "gold_legal_id": str(row.get("gold_legal_id", "")).strip(),
                "relevance_grade": max(0, min(3, grade)),
            }
        )
        query_texts[query_id] = query
        # Primary bucket from first row's source
        if query_id not in query_source:
            query_source[query_id] = gold_source

    return qrels_by_query, query_texts, query_source


def source_bucket(gold_source: str) -> str:
    s = gold_source.lower()
    if "faq" in s:
        return "FAQ"
    if "refund" in s:
        return "Refund"
    if "term" in s:
        return "Terms"
    return "Other"


def doc_matches_qrel(doc, qrel: Dict[str, Any]) -> bool:
    import re

    source = doc.metadata.get("source", "").lower()
    if qrel["gold_source"] not in source:
        return False

    gold_faq_id = qrel.get("gold_faq_id", "")
    gold_legal_id = qrel.get("gold_legal_id", "")
    if gold_faq_id:
        if doc.metadata.get("faq_id") != gold_faq_id:
            return False
    elif gold_legal_id:
        if doc.metadata.get("legal_id") != gold_legal_id:
            return False
    elif qrel["gold_section_regex"]:
        section = str(doc.metadata.get("section", ""))
        heading = str(doc.metadata.get("legal_heading", ""))
        target = f"{section}\n{heading}\n{doc.page_content[:800]}"
        try:
            if not re.search(qrel["gold_section_regex"], target, re.IGNORECASE):
                return False
        except re.error:
            return False

    if qrel["gold_keywords"]:
        text = doc.page_content.lower()
        keywords = qrel["gold_keywords"]
        if gold_faq_id or gold_legal_id:
            hits = sum(1 for k in keywords if k in text)
            if hits < min(2, len(keywords)):
                return False
        elif not all(k in text for k in keywords):
            return False
    return True


def evaluate_query(docs: List[Any], qrels: List[Dict[str, Any]]) -> Dict[str, float]:
    if not qrels:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    matched = set()
    per_rank_bin: List[int] = []
    per_rank_grade: List[int] = []

    for doc in docs:
        best_idx = None
        best_grade = -1
        for idx, qrel in enumerate(qrels):
            if idx in matched:
                continue
            if doc_matches_qrel(doc, qrel):
                if qrel["relevance_grade"] > best_grade:
                    best_grade = qrel["relevance_grade"]
                    best_idx = idx
        if best_idx is not None:
            matched.add(best_idx)
            per_rank_bin.append(1)
            per_rank_grade.append(best_grade)
        else:
            per_rank_bin.append(0)
            per_rank_grade.append(0)

    k = len(docs)
    relevant_retrieved = sum(per_rank_bin)
    total_relevant = len(qrels)
    precision = relevant_retrieved / k if k else 0.0
    recall = relevant_retrieved / total_relevant if total_relevant else 0.0

    mrr = 0.0
    for rank, rel in enumerate(per_rank_bin, start=1):
        if rel == 1:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    for rank, g in enumerate(per_rank_grade, start=1):
        if g > 0:
            dcg += (2**g - 1) / math.log2(rank + 1)
    ideal = sorted((q["relevance_grade"] for q in qrels), reverse=True)[:k]
    idcg = 0.0
    for rank, g in enumerate(ideal, start=1):
        if g > 0:
            idcg += (2**g - 1) / math.log2(rank + 1)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "ndcg_at_k": ndcg,
    }


def load_store(persist_dir: Path) -> Chroma:
    import json

    meta_path = persist_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        embedding_type = meta.get("embedding_type", "ollama")
        model_key = meta.get("model_key")
    else:
        embedding_type, model_key = "ollama", "qwen3-0.6b"

    es = EmbeddingService(
        embedding_type=EmbeddingType(embedding_type),
        model_key=model_key,
    )
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=es.embeddings,
    )


def run_mode(
    mode: str,
    vectorstore: Chroma,
    bm25: BM25Index,
    query_texts: Dict[str, str],
    qrels_by_query: Dict[str, List[Dict[str, Any]]],
    query_source: Dict[str, str],
    top_k: int,
    dense_weight: float,
    sparse_weight: float,
    policy_only: bool = False,
) -> pd.DataFrame:
    from app.services.query_intent import should_use_hybrid_retrieval

    rows = []
    for query_id, query in query_texts.items():
        t0 = time.time()
        use_hybrid = mode != "dense"
        if use_hybrid:
            use_hybrid = should_use_hybrid_retrieval(query, policy_only)

        if not use_hybrid:
            docs = vectorstore.similarity_search(query, k=top_k)
        else:
            docs = hybrid_search(
                vectorstore=vectorstore,
                bm25_index=bm25,
                query=query,
                fetch_k=top_k,
                use_hybrid=True,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        elapsed = time.time() - t0
        metrics = evaluate_query(docs, qrels_by_query[query_id])
        rows.append(
            {
                "mode": mode,
                "query_id": query_id,
                "query": query,
                "bucket": source_bucket(query_source[query_id]),
                "response_time": elapsed,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for (mode, bucket), g in df.groupby(["mode", "bucket"]):
        groups.append(
            {
                "mode": mode,
                "bucket": bucket,
                "n": len(g),
                "mrr": g["mrr"].mean(),
                "recall_at_k": g["recall_at_k"].mean(),
                "ndcg_at_k": g["ndcg_at_k"].mean(),
                "precision_at_k": g["precision_at_k"].mean(),
                "hit_rate": (g["recall_at_k"] > 0).mean(),
                "latency_s": g["response_time"].mean(),
            }
        )
    # overall per mode
    for mode, g in df.groupby("mode"):
        groups.append(
            {
                "mode": mode,
                "bucket": "ALL",
                "n": len(g),
                "mrr": g["mrr"].mean(),
                "recall_at_k": g["recall_at_k"].mean(),
                "ndcg_at_k": g["ndcg_at_k"].mean(),
                "precision_at_k": g["precision_at_k"].mean(),
                "hit_rate": (g["recall_at_k"] > 0).mean(),
                "latency_s": g["response_time"].mean(),
            }
        )
    return pd.DataFrame(groups).sort_values(["bucket", "mode"])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dense vs hybrid(50/50) retrieval, especially Terms/Refund MRR."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/benchmark_results/hr_qrels_full_first_pass_for_review.csv",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/chroma_db/ollama_qwen3-0.6b",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--sparse-weight", type=float, default=0.5)
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="Apply hybrid only when query looks like policy/legal",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Compare dense, 50/50, 30/70, and policy-only 30/70",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_results",
    )
    args = parser.parse_args()

    qrels_by_query, query_texts, query_source = load_qrels(Path(args.dataset_path))
    persist_dir = Path(args.persist_dir)
    print(f"Loading store: {persist_dir}")
    vs = load_store(persist_dir)
    bm25 = BM25Index.from_vectorstore(vs)
    print(f"Queries={len(query_texts)} chunks={len(bm25.documents)}")

    configs = []
    if args.sweep:
        configs = [
            ("dense", 1.0, 0.0, False),
            ("hybrid_50_50", 0.5, 0.5, False),
            ("hybrid_30_70", 0.3, 0.7, False),
            ("hybrid_30_70_policyish_skip_refund", 0.3, 0.7, True),
            ("hybrid_50_50_policyish_skip_refund", 0.5, 0.5, True),
        ]
    else:
        configs = [
            ("dense", 1.0, 0.0, False),
            (
                "hybrid",
                args.dense_weight,
                args.sparse_weight,
                args.policy_only,
            ),
        ]

    frames = []
    for mode_name, dw, sw, policy_only in configs:
        print(
            f"Running mode={mode_name} dense={dw} sparse={sw} "
            f"policy_only={policy_only} ..."
        )
        if mode_name == "dense":
            df = run_mode(
                mode="dense",
                vectorstore=vs,
                bm25=bm25,
                query_texts=query_texts,
                qrels_by_query=qrels_by_query,
                query_source=query_source,
                top_k=args.top_k,
                dense_weight=dw,
                sparse_weight=sw,
                policy_only=False,
            )
        else:
            df = run_mode(
                mode=mode_name,
                vectorstore=vs,
                bm25=bm25,
                query_texts=query_texts,
                qrels_by_query=qrels_by_query,
                query_source=query_source,
                top_k=args.top_k,
                dense_weight=dw,
                sparse_weight=sw,
                policy_only=policy_only,
            )
            df["mode"] = mode_name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    summary = summarize(all_df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir / f"hybrid_eval_detail_{stamp}.csv"
    summary_path = out_dir / f"hybrid_eval_summary_{stamp}.csv"
    all_df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== MRR by corpus bucket ===")
    focus = summary[summary["bucket"].isin(["Terms", "Refund", "FAQ", "ALL"])]
    print(
        focus[["mode", "bucket", "n", "mrr", "recall_at_k", "hit_rate", "ndcg_at_k"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"\nWrote {summary_path}")
    print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()
