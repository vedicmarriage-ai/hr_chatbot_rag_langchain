import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import docx2txt
from openai import OpenAI


FAQ_SPLIT = re.compile(r"\n(?=Q\d+\.)")
QUESTION_RE = re.compile(r"^Q(\d+)\.\s*(.+)")

REQUIRED_ITEM_KEYS = {
    "item_id",
    "q_no",
    "query",
    "gold_keywords",
    "relevance_grade",
    "notes",
}


def parse_faq_blocks(faq_path: Path) -> List[Dict[str, str]]:
    text = docx2txt.process(str(faq_path)).strip()
    blocks = FAQ_SPLIT.split(text)
    section = "FAQ"
    parsed = []
    item_idx = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        q_match = QUESTION_RE.match(block)
        if q_match:
            q_no = q_match.group(1)
            question_line = q_match.group(2).strip()
            parsed.append(
                {
                    "item_id": f"{item_idx + 1:03d}",
                    "q_no": q_no,
                    "question": question_line,
                    "section": section,
                }
            )
            item_idx += 1
        else:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if lines:
                section = lines[-1]

    return parsed


def make_prompt(batch: List[Dict[str, str]]) -> str:
    payload = [
        {
            "item_id": b["item_id"],
            "q_no": b["q_no"],
            "question": b["question"],
            "section": b["section"],
        }
        for b in batch
    ]
    return f"""Create first-pass qrels rows for retrieval evaluation.
Return ONLY a JSON array of objects. Do not include markdown.

Each object must include:
- item_id (string, echo exactly)
- q_no (string, echo exactly)
- query (natural user phrasing)
- gold_keywords (pipe-separated string, 2-6 terms)
- relevance_grade (integer 2 or 3)
- notes (short text)

Rules:
- Keep intent equivalent to the original FAQ question.
- gold_keywords should be concrete terms likely present in evidence chunk.
- Use grade 3 for direct evidence, 2 for broader wording.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def extract_json_array(text: str) -> List[Dict[str, str]]:
    # direct parse first (for strict JSON mode outputs)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("items", "rows", "data", "qrels"):
                val = obj.get(key)
                if isinstance(val, list):
                    return val
    except Exception:
        pass

    # fallback: try to slice out an array from mixed output
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output does not contain a JSON array.")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("Model output JSON root is not a list.")
    return data


def validate_item_shape(item: Dict[str, str]) -> bool:
    return isinstance(item, dict) and REQUIRED_ITEM_KEYS.issubset(set(item.keys()))


def fallback_rows_for_batch(
    batch: List[Dict[str, str]], source_filename: str
) -> List[Dict[str, str]]:
    rows = []
    for q in batch:
        qid = f"FAQ-{q['item_id']}"
        keywords = "|".join(
            [w.lower() for w in re.findall(r"[A-Za-z]{4,}", q["question"])[:5]]
        )
        rows.append(
            {
                "query_id": qid,
                "query": q["question"],
                "gold_source": source_filename,
                "gold_section_regex": rf"Q{q['q_no']}\.",
                "gold_keywords": keywords or "faq|question",
                "relevance_grade": "3",
                "notes": "Fallback deterministic row",
            }
        )
    return rows


def call_openai_with_retries(
    client: OpenAI,
    model_name: str,
    prompt: str,
    retries: int = 3,
) -> List[Dict[str, str]]:
    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You output strictly valid JSON arrays only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
            data = extract_json_array(text)
            if not all(validate_item_shape(item) for item in data):
                raise ValueError("One or more items missing required keys.")
            return data
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
    raise RuntimeError(f"OpenAI generation failed after retries: {last_error}")


def generate_first_pass_rows(
    parsed_questions: List[Dict[str, str]],
    source_filename: str,
    client: OpenAI,
    model_name: str,
) -> List[Dict[str, str]]:
    rows = []
    batch_size = 10

    for i in range(0, len(parsed_questions), batch_size):
        batch = parsed_questions[i : i + batch_size]
        prompt = make_prompt(batch)
        try:
            items = call_openai_with_retries(client, model_name, prompt)
        except Exception:
            rows.extend(fallback_rows_for_batch(batch, source_filename))
            continue

        q_by_item = {q["item_id"]: q for q in batch}
        for item in items:
            item_id = str(item.get("item_id", "")).strip()
            if item_id not in q_by_item:
                continue
            q = q_by_item[item_id]
            q_no = q["q_no"]
            query_id = f"FAQ-{item_id}"
            rows.append(
                {
                    "query_id": query_id,
                    "query": str(item.get("query", q["question"])).strip()
                    or q["question"],
                    "gold_source": source_filename,
                    "gold_section_regex": rf"Q{q_no}\.",
                    "gold_keywords": str(item.get("gold_keywords", "")).strip()
                    or "faq|question",
                    "relevance_grade": str(item.get("relevance_grade", 3)),
                    "notes": str(
                        item.get("notes", f"Auto-generated from FAQ Q{q_no}")
                    ).strip(),
                }
            )

    # fill any skipped IDs deterministically
    existing_ids = {r["query_id"] for r in rows}
    for q in parsed_questions:
        qid = f"FAQ-{q['item_id']}"
        if qid in existing_ids:
            continue
        rows.extend(fallback_rows_for_batch([q], source_filename))

    rows.sort(key=lambda r: r["query_id"])
    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query",
        "gold_source",
        "gold_section_regex",
        "gold_keywords",
        "relevance_grade",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate first-pass qrels from FAQ using an OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--faq-path",
        default="data/docs/Text_documents/FAQ.docx",
        help="Path to FAQ.docx",
    )
    parser.add_argument(
        "--output-path",
        default="data/benchmark_results/hr_qrels_from_faq_openai_first_pass.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4o-mini",
        help="OpenAI model name",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required.")

    faq_path = Path(args.faq_path)
    if not faq_path.exists():
        raise FileNotFoundError(f"FAQ not found at {faq_path}")

    parsed_questions = parse_faq_blocks(faq_path)
    if not parsed_questions:
        raise ValueError("No FAQ questions found in document.")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    rows = generate_first_pass_rows(
        parsed_questions=parsed_questions,
        source_filename=faq_path.name,
        client=client,
        model_name=args.model_name,
    )
    write_csv(rows, Path(args.output_path))
    print(f"Generated {len(rows)} rows at: {args.output_path}")
    print("Next: review/edit, then run validate_qrels.py.")


if __name__ == "__main__":
    main()
