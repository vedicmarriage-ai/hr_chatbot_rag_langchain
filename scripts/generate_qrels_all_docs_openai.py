"""
Build one master qrels CSV for manual review:
- Reuses existing FAQ OpenAI qrels (no re-generation cost)
- Generates first-pass qrels for legal/policy docs from indexed chunks
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.core.config import get_settings
from app.services.document_service import DocumentService
from scripts.generate_qrels_from_faq_openai import extract_json_array, write_csv

settings = get_settings()

LEGAL_SOURCES = {
    "refund": "VedicMarriage_Refund_Cancellation_Policy.docx",
    "terms": "terms and condition of vedicmarriage@ai.docx",
}

REQUIRED_KEYS = {"item_id", "query", "gold_keywords", "relevance_grade", "notes"}


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_section_regex(content: str, section: str) -> str:
    first_line = content.strip().split("\n")[0].strip()

    q_match = re.match(r"^(Q\d+\.)", first_line)
    if q_match:
        return re.escape(q_match.group(1))

    clause_match = re.match(r"^(\d+\.\d+\.)", first_line)
    if clause_match:
        return re.escape(clause_match.group(1))

    section_match = re.match(r"^(\d+\.\s+[A-Za-z])", first_line)
    if section_match:
        return re.escape(first_line.split("\n")[0][:40])

    if section and section not in ("Introduction", "FAQ", "?"):
        return re.escape(section[:80])

    words = re.findall(r"[A-Za-z]{5,}", content[:250])[:4]
    if words:
        return "|".join(re.escape(w) for w in words)
    return "."


def chunk_to_evidence_units(chunks) -> List[Dict[str, str]]:
    units = []
    counters = {"refund": 0, "terms": 0}

    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "")).name
        if source not in LEGAL_SOURCES.values():
            continue

        if len(chunk.page_content.strip()) < 100:
            continue

        prefix = "REFUND" if "refund" in source.lower() else "TERMS"
        counters["refund" if prefix == "REFUND" else "terms"] += 1
        idx = counters["refund" if prefix == "REFUND" else "terms"]

        section = str(chunk.metadata.get("section", "Introduction"))
        content = chunk.page_content.strip()

        units.append(
            {
                "item_id": f"{prefix}-{idx:03d}",
                "source_filename": source,
                "section": section,
                "excerpt": content[:900],
                "gold_section_regex": build_section_regex(content, section),
            }
        )
    return units


def make_legal_prompt(batch: List[Dict[str, str]]) -> str:
    payload = [
        {
            "item_id": u["item_id"],
            "section": u["section"],
            "source": u["source_filename"],
            "excerpt": u["excerpt"][:700],
        }
        for u in batch
    ]
    return f"""Create first-pass qrels for HR chatbot retrieval evaluation.
Return ONLY JSON: {{"items": [ ... ]}}

For each input evidence chunk, output 1 or 2 objects with:
- item_id (echo exactly; append -a / -b if two queries for same chunk)
- query (natural user question answerable from excerpt)
- gold_keywords (pipe-separated, 2-6 concrete terms from excerpt)
- relevance_grade (2 or 3)
- notes (short)

Rules:
- Do not invent facts outside excerpt.
- Grade 3 = direct answer in this chunk; 2 = related but broader.
- Questions should sound like real users asking HR/policy questions.

Input chunks:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def fallback_legal_row(unit: Dict[str, str], suffix: str = "") -> Dict[str, str]:
    words = re.findall(r"[A-Za-z]{4,}", unit["excerpt"])[:6]
    keywords = "|".join(w.lower() for w in words[:5]) or "policy|terms"
    section = unit["section"]
    query = f"What does the {section} section say?" if section != "Introduction" else f"What does {unit['source_filename']} say?"
    return {
        "query_id": f"{unit['item_id']}{suffix}",
        "query": query,
        "gold_source": unit["source_filename"],
        "gold_section_regex": unit["gold_section_regex"],
        "gold_keywords": keywords,
        "relevance_grade": "2",
        "notes": "Fallback deterministic row",
    }


def _unit_for_item_id(batch: List[Dict[str, str]], item_id: str) -> Optional[Dict[str, str]]:
    base = re.sub(r"-[ab]$", "", item_id)
    for unit in batch:
        if item_id == unit["item_id"] or item_id.startswith(unit["item_id"]):
            return unit
        if base == unit["item_id"]:
            return unit
    return None


def generate_legal_rows(
    units: List[Dict[str, str]], client: OpenAI, model_name: str
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    batch_size = 5

    for i in range(0, len(units), batch_size):
        batch = units[i : i + batch_size]
        prompt = make_legal_prompt(batch)
        batch_rows: List[Dict[str, str]] = []
        covered_units: set[str] = set()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You output strictly valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
            obj = json.loads(text)
            items = obj.get("items", []) if isinstance(obj, dict) else extract_json_array(text)
        except Exception:
            items = []

        per_unit_count: Dict[str, int] = {}
        for item in items:
            if not isinstance(item, dict) or not REQUIRED_KEYS.issubset(item.keys()):
                continue
            item_id = str(item["item_id"]).strip()
            unit = _unit_for_item_id(batch, item_id)
            if not unit:
                continue

            count = per_unit_count.get(unit["item_id"], 0)
            suffix = "" if count == 0 else chr(ord("a") + count - 1)
            qid = f"{unit['item_id']}{'-' + suffix if suffix else ''}"
            per_unit_count[unit["item_id"]] = count + 1
            covered_units.add(unit["item_id"])

            batch_rows.append(
                {
                    "query_id": qid,
                    "query": str(item["query"]).strip(),
                    "gold_source": unit["source_filename"],
                    "gold_section_regex": unit["gold_section_regex"],
                    "gold_keywords": str(item["gold_keywords"]).strip(),
                    "relevance_grade": str(item["relevance_grade"]),
                    "notes": str(item.get("notes", "OpenAI legal draft")).strip(),
                }
            )

        for unit in batch:
            if unit["item_id"] not in covered_units:
                batch_rows.append(fallback_legal_row(unit))

        rows.extend(batch_rows)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate full qrels draft (FAQ + legal docs) for manual review."
    )
    parser.add_argument(
        "--faq-qrels",
        default="data/benchmark_results/hr_qrels_from_faq_openai_first_pass.csv",
        help="Existing FAQ qrels CSV to include without regenerating",
    )
    parser.add_argument(
        "--output-path",
        default="data/benchmark_results/hr_qrels_full_first_pass_for_review.csv",
        help="Combined output CSV for manual audit",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4o-mini",
        help="OpenAI model for legal/policy qrels generation",
    )
    parser.add_argument(
        "--docs-dir",
        default=str(settings.DOCS_DIR),
        help="Directory containing HR documents",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required. Load .env.dev first.")

    faq_rows = load_csv_rows(Path(args.faq_qrels))
    print(f"Loaded {len(faq_rows)} FAQ qrels from {args.faq_qrels}")

    ds = DocumentService()
    docs = ds.load_documents(args.docs_dir)
    chunks = ds.split_documents(docs)
    units = chunk_to_evidence_units(chunks)
    print(f"Found {len(units)} legal/policy evidence units to process")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    legal_rows = generate_legal_rows(units, client, args.model_name)
    print(f"Generated {len(legal_rows)} legal/policy qrels rows")

    all_rows = faq_rows + legal_rows
    all_rows.sort(key=lambda r: (r["gold_source"], r["query_id"]))

    out = Path(args.output_path)
    write_csv(all_rows, out)
    print(f"Wrote {len(all_rows)} total rows to {out}")
    print("Next steps:")
    print(f"  1) Manually review/edit {out}")
    print(f"  2) python scripts/validate_qrels.py --dataset-path {out}")


if __name__ == "__main__":
    main()
