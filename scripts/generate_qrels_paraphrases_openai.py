"""
Expand a qrels CSV with natural-language paraphrase queries via OpenAI.

For each base row (one query_id), adds 2-3 paraphrased query rows that keep the
same gold_source, gold_section_regex, gold_keywords, and relevance_grade.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set

from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env.dev")
load_dotenv(PROJECT_ROOT / ".env")

FIELDNAMES = [
    "query_id",
    "query",
    "gold_source",
    "gold_section_regex",
    "gold_keywords",
    "relevance_grade",
    "notes",
]

PARAPHRASE_SUFFIX_RE = re.compile(r"-p\d+$", re.IGNORECASE)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def is_base_row(row: Dict[str, str]) -> bool:
    query_id = (row.get("query_id") or "").strip()
    notes = (row.get("notes") or "").strip().lower()
    if not query_id:
        return False
    if PARAPHRASE_SUFFIX_RE.search(query_id):
        return False
    if notes.startswith("paraphrase of"):
        return False
    return True


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def make_prompt(batch: List[Dict[str, str]], count: int) -> str:
    payload = [
        {
            "item_id": row["query_id"],
            "original_query": row["query"],
            "gold_source": row["gold_source"],
        }
        for row in batch
    ]
    return f"""Create natural user-query paraphrases for retrieval benchmark evaluation.
Return ONLY JSON: {{"items": [ ... ]}}

For each input item, output exactly {count} paraphrases with:
- item_id (echo exactly)
- paraphrases (array of {count} strings)

Rules:
- Keep the same intent and answerability as original_query.
- Use realistic HR/user wording users might type in chat.
- Vary vocabulary: e.g. register -> join/sign up/create account; website -> platform/site/app.
- Do NOT copy the original query verbatim.
- Paraphrases must be distinct from each other and from original_query.
- Keep questions about Vedic Marriage / matrimony context when relevant.
- Shorter and longer variants are both fine.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def simple_fallback_paraphrases(query: str, count: int) -> List[str]:
    replacements = [
        (r"\bregister\b", "join"),
        (r"\bregistration\b", "signing up"),
        (r"\bwebsite\b", "platform"),
        (r"\bHow can I\b", "How do I"),
        (r"\bWhat is\b", "Can you explain"),
        (r"\bHow do I\b", "What's the way to"),
    ]
    results: List[str] = []
    for i in range(count):
        text = query
        pair = replacements[i % len(replacements)]
        text = re.sub(pair[0], pair[1], text, count=1, flags=re.IGNORECASE)
        if normalize_query(text) != normalize_query(query):
            results.append(text)
    if not results:
        results = [f"{query} (alternate wording {i + 1})" for i in range(count)]
    return results[:count]


def call_openai_batch(
    client: OpenAI,
    model_name: str,
    batch: List[Dict[str, str]],
    count: int,
) -> Dict[str, List[str]]:
    prompt = make_prompt(batch, count)
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
    items = obj.get("items", []) if isinstance(obj, dict) else []

    out: Dict[str, List[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id", "")).strip()
        paraphrases = item.get("paraphrases", [])
        if not item_id or not isinstance(paraphrases, list):
            continue
        cleaned: List[str] = []
        for p in paraphrases:
            p = str(p).strip()
            if p:
                cleaned.append(p)
        if cleaned:
            out[item_id] = cleaned
    return out


def build_paraphrase_rows(
    base_row: Dict[str, str],
    paraphrases: List[str],
    count: int,
    existing_queries: Set[str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    base_id = base_row["query_id"].strip()
    base_norm = normalize_query(base_row["query"])

    for i, phrase in enumerate(paraphrases[:count], start=1):
        norm = normalize_query(phrase)
        if not norm or norm == base_norm or norm in existing_queries:
            continue

        rows.append(
            {
                "query_id": f"{base_id}-p{i}",
                "query": phrase.strip(),
                "gold_source": base_row["gold_source"],
                "gold_section_regex": base_row.get("gold_section_regex", ""),
                "gold_keywords": base_row["gold_keywords"],
                "relevance_grade": base_row["relevance_grade"],
                "notes": f"Paraphrase of {base_id}",
            }
        )
        existing_queries.add(norm)

    return rows


def generate_paraphrases(
    base_rows: List[Dict[str, str]],
    client: OpenAI,
    model_name: str,
    count: int,
    batch_size: int,
) -> List[Dict[str, str]]:
    paraphrase_rows: List[Dict[str, str]] = []
    existing_queries = {normalize_query(r["query"]) for r in base_rows}

    for i in range(0, len(base_rows), batch_size):
        batch = base_rows[i : i + batch_size]
        try:
            by_id = call_openai_batch(client, model_name, batch, count)
        except Exception:
            by_id = {}

        for row in batch:
            qid = row["query_id"]
            phrases = by_id.get(qid, [])
            if len(phrases) < count:
                phrases = phrases + simple_fallback_paraphrases(
                    row["query"], count - len(phrases)
                )
            paraphrase_rows.extend(
                build_paraphrase_rows(row, phrases, count, existing_queries)
            )

    return paraphrase_rows


def sort_key(row: Dict[str, str]):
    qid = row["query_id"]
    m = re.match(r"^(.+?)(-p\d+)?$", qid, re.IGNORECASE)
    base = m.group(1) if m else qid
    suffix = m.group(2) or ""
    suffix_num = int(suffix[2:]) if suffix else 0
    return (row.get("gold_source", ""), base, suffix_num)


def main():
    parser = argparse.ArgumentParser(
        description="Add OpenAI-generated paraphrase qrels rows to an existing CSV."
    )
    parser.add_argument(
        "--input-path",
        default="data/benchmark_results/hr_qrels_full_first_pass_for_review.csv",
        help="Input qrels CSV",
    )
    parser.add_argument(
        "--output-path",
        default="data/benchmark_results/hr_qrels_with_paraphrases.csv",
        help="Output CSV with originals + paraphrases",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4o-mini",
        help="OpenAI model name",
    )
    parser.add_argument(
        "--paraphrases-per-query",
        type=int,
        default=3,
        help="Number of paraphrase rows to add per base query (1-10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Number of base queries per OpenAI request",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only write paraphrase rows (no originals)",
    )
    args = parser.parse_args()
    if not 1 <= args.paraphrases_per_query <= 10:
        parser.error("--paraphrases-per-query must be between 1 and 10.")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required. Set it in .env.dev or export it in your shell."
        )

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    all_rows = load_rows(input_path)
    base_rows = [r for r in all_rows if is_base_row(r)]
    if not base_rows:
        raise ValueError("No base rows found to paraphrase.")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    print(f"Loaded {len(all_rows)} rows; paraphrasing {len(base_rows)} base queries...")
    paraphrase_rows = generate_paraphrases(
        base_rows=base_rows,
        client=client,
        model_name=args.model_name,
        count=args.paraphrases_per_query,
        batch_size=args.batch_size,
    )

    if args.base_only:
        output_rows = paraphrase_rows
    else:
        output_rows = all_rows + paraphrase_rows

    output_rows.sort(key=sort_key)
    write_rows(output_rows, Path(args.output_path))

    print(f"Added {len(paraphrase_rows)} paraphrase rows.")
    print(f"Wrote {len(output_rows)} total rows to {args.output_path}")
    print("Next: review, then run validate_qrels.py on the output file.")


if __name__ == "__main__":
    main()
