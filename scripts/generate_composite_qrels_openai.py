"""
Generate multi-chunk composite qrels with OpenAI paraphrases.

Reads query + evidence definitions from a JSON config (no hardcoded queries).
Each composite query_id gets multiple qrel rows (one per chunk) and optional
paraphrase variants that duplicate the same chunk set.

Example:
  python scripts/generate_composite_qrels_openai.py \\
    --config data/benchmark_results/composite_queries.json \\
    --merge-into data/benchmark_results/hr_qrels_with_paraphrases.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env.dev")
load_dotenv(PROJECT_ROOT / ".env")

from app.core.config import get_settings
from app.services.document_service import DocumentService
from scripts.patch_faq_qrels import _build_section_regex

FIELDNAMES = [
    "query_id",
    "query",
    "gold_source",
    "gold_faq_id",
    "gold_section_regex",
    "gold_keywords",
    "relevance_grade",
    "notes",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def default_keywords(text: str) -> str:
    words = re.findall(r"[A-Za-z]{4,}", text[:400])[:5]
    return "|".join(w.lower() for w in words) or "policy"


def build_corpus_index() -> Dict[str, Any]:
    settings = get_settings()
    ds = DocumentService()
    docs = ds.load_documents(str(settings.DOCS_DIR))
    chunks = ds.split_documents(docs)

    faq_by_id: Dict[str, Any] = {}
    legal_chunks: List[Any] = []

    for chunk in chunks:
        faq_id = chunk.metadata.get("faq_id")
        if faq_id:
            faq_by_id[str(faq_id)] = chunk
        else:
            legal_chunks.append(chunk)

    return {"faq_by_id": faq_by_id, "legal_chunks": legal_chunks}


def resolve_chunk_spec(
    spec: Dict[str, Any], corpus: Dict[str, Any]
) -> Dict[str, str]:
    if "ref" in spec:
        chunk = corpus["faq_by_id"].get(spec["ref"])
        if not chunk:
            raise ValueError(f"Unknown FAQ ref: {spec['ref']}")
        source = Path(chunk.metadata.get("source", "")).name
        faq_id = str(chunk.metadata.get("faq_id", ""))
        section = str(chunk.metadata.get("section", ""))
        faq_question = str(chunk.metadata.get("faq_question", ""))
        keywords = spec.get("keywords") or default_keywords(chunk.page_content)
        return {
            "gold_source": source,
            "gold_faq_id": faq_id,
            "gold_section_regex": _build_section_regex(section, faq_question),
            "gold_keywords": keywords,
            "relevance_grade": str(spec.get("relevance_grade", 3)),
            "notes": str(spec.get("notes", f"Composite evidence: {faq_id}")),
        }

    match = spec.get("match")
    if not match:
        raise ValueError(f"Chunk spec must include 'ref' or 'match': {spec}")

    source_name = match["source"]
    content_pattern = match["content_match"]
    chunk = None
    for candidate in corpus["legal_chunks"]:
        src = Path(candidate.metadata.get("source", "")).name
        if src != source_name:
            continue
        if re.search(content_pattern, candidate.page_content, re.IGNORECASE):
            chunk = candidate
            break
    if not chunk:
        raise ValueError(
            f"No chunk matched source={source_name} content_match={content_pattern}"
        )

    section = str(chunk.metadata.get("section", ""))
    keywords = spec.get("keywords") or default_keywords(chunk.page_content)
    regex = match.get("section_regex") or content_pattern
    return {
        "gold_source": source_name,
        "gold_faq_id": "",
        "gold_section_regex": regex,
        "gold_keywords": keywords,
        "relevance_grade": str(spec.get("relevance_grade", 3)),
        "notes": str(spec.get("notes", "Composite legal evidence")),
    }


def make_paraphrase_prompt(query: str, chunk_summaries: List[str], count: int) -> str:
    return f"""Generate natural user questions for an HR/matrimony chatbot retrieval benchmark.
Return ONLY JSON: {{"paraphrases": ["...", ...]}}

The question requires MULTIPLE pieces of evidence from our docs (composite intent).
Keep the same combined intent as the original — do not narrow to only one sub-topic.

Original query:
{query}

Evidence topics involved:
{json.dumps(chunk_summaries, ensure_ascii=False, indent=2)}

Rules:
- Output exactly {count} paraphrases.
- Vary wording (register/join/sign up, free/cost, plans/tiers, login/password, refund/cancel).
- Do not copy the original verbatim.
- Each paraphrase must still need the same multi-part answer.

Examples of good composite paraphrases for registration+cost:
- "How can I sign up and does it cost anything?"
- "Is joining Vedic Marriage free and how do I start?"
"""


def simple_paraphrases(query: str, count: int) -> List[str]:
    replacements = [
        (r"\bregister\b", "sign up"),
        (r"\bregistration\b", "joining"),
        (r"\bfree\b", "cost anything"),
        (r"\bplans\b", "membership options"),
        (r"\blog in\b", "access my account"),
        (r"\brefund\b", "money back"),
    ]
    results = []
    for i in range(count):
        text = query
        pat, repl = replacements[i % len(replacements)]
        text = re.sub(pat, repl, text, count=1, flags=re.IGNORECASE)
        if text.lower() != query.lower():
            results.append(text)
    while len(results) < count:
        results.append(f"{query} (variant {len(results) + 1})")
    return results[:count]


def generate_paraphrases(
    client: OpenAI,
    model_name: str,
    query: str,
    chunk_summaries: List[str],
    count: int,
) -> List[str]:
    prompt = make_paraphrase_prompt(query, chunk_summaries, count)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You output strictly valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        text = response.choices[0].message.content or ""
        obj = json.loads(text)
        phrases = obj.get("paraphrases", [])
        cleaned = [str(p).strip() for p in phrases if str(p).strip()]
        if len(cleaned) >= count:
            return cleaned[:count]
    except Exception:
        pass
    return simple_paraphrases(query, count)


def build_composite_rows(
    composite: Dict[str, Any],
    corpus: Dict[str, Any],
    paraphrases: List[str],
) -> List[Dict[str, str]]:
    base_id = composite["id"]
    canonical_query = composite["query"]
    chunk_specs = composite["chunks"]

    resolved = [resolve_chunk_spec(spec, corpus) for spec in chunk_specs]
    summaries = [r["notes"] for r in resolved]

    query_variants = [(base_id, canonical_query)]
    for i, phrase in enumerate(paraphrases, start=1):
        query_variants.append((f"{base_id}-p{i}", phrase))

    rows: List[Dict[str, str]] = []
    for qid, qtext in query_variants:
        for chunk_row in resolved:
            rows.append(
                {
                    "query_id": qid,
                    "query": qtext,
                    "gold_source": chunk_row["gold_source"],
                    "gold_faq_id": chunk_row["gold_faq_id"],
                    "gold_section_regex": chunk_row["gold_section_regex"],
                    "gold_keywords": chunk_row["gold_keywords"],
                    "relevance_grade": chunk_row["relevance_grade"],
                    "notes": f"Composite {base_id}: {chunk_row['notes']}",
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-chunk composite qrels with paraphrases from JSON config."
    )
    parser.add_argument(
        "--config",
        default="data/benchmark_results/composite_queries.json",
        help="JSON file listing composite queries and chunk refs",
    )
    parser.add_argument(
        "--output-path",
        default="data/benchmark_results/hr_qrels_composite.csv",
        help="Output CSV for composite rows only",
    )
    parser.add_argument(
        "--merge-into",
        default="",
        help="Optional: append composite rows to an existing qrels CSV",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4o-mini",
        help="OpenAI model for paraphrases",
    )
    parser.add_argument(
        "--paraphrases-per-query",
        type=int,
        default=0,
        help="Override paraphrase count from config (0 = use config)",
    )
    parser.add_argument(
        "--no-paraphrases",
        action="store_true",
        help="Skip OpenAI paraphrases; emit canonical queries only",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    paraphrase_count = args.paraphrases_per_query or int(
        config.get("paraphrases_per_query", 3)
    )
    if not 0 <= paraphrase_count <= 10:
        parser.error("--paraphrases-per-query must be between 0 and 10.")

    corpus = build_corpus_index()
    composites = config.get("composites", [])
    if not composites:
        raise ValueError(f"No composites found in {config_path}")

    client = None
    if not args.no_paraphrases and paraphrase_count > 0:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for paraphrases.")
        base_url = os.getenv("OPENAI_BASE_URL")
        client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )

    all_rows: List[Dict[str, str]] = []
    for composite in composites:
        summaries = [
            str(spec.get("notes") or spec.get("ref") or spec.get("match", ""))
            for spec in composite["chunks"]
        ]
        if args.no_paraphrases or paraphrase_count == 0:
            paraphrases = []
        else:
            paraphrases = generate_paraphrases(
                client,
                args.model_name,
                composite["query"],
                summaries,
                paraphrase_count,
            )
        all_rows.extend(build_composite_rows(composite, corpus, paraphrases))
        print(
            f"{composite['id']}: {len(composite['chunks'])} chunks × "
            f"{1 + len(paraphrases)} queries = "
            f"{len(composite['chunks']) * (1 + len(paraphrases))} rows"
        )

    write_csv_rows(Path(args.output_path), all_rows)
    print(f"Wrote {len(all_rows)} composite rows to {args.output_path}")

    if args.merge_into:
        merge_path = Path(args.merge_into)
        existing = load_csv_rows(merge_path)
        composite_ids = {c["id"] for c in composites}

        def is_old_composite_row(row: Dict[str, str]) -> bool:
            qid = row.get("query_id", "")
            return any(qid == cid or qid.startswith(f"{cid}-p") for cid in composite_ids)

        filtered = [r for r in existing if not is_old_composite_row(r)]
        merged = filtered + all_rows
        write_csv_rows(merge_path, merged)
        print(
            f"Merged into {merge_path}: {len(filtered)} existing + "
            f"{len(all_rows)} composite = {len(merged)} total"
        )
        print("Next: python scripts/validate_qrels.py --dataset-path", merge_path)


if __name__ == "__main__":
    main()
