import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict

import docx2txt

# Add project root
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.services.llm_service import LLMService


FAQ_SPLIT = re.compile(r"\n(?=Q\d+\.)")
QUESTION_RE = re.compile(r"^Q(\d+)\.\s*(.+)")


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
    return f"""You are creating first-pass qrels rows for retrieval evaluation.
Return ONLY a JSON array (no markdown).

For each item, return object with keys:
- q_no (string)
- item_id (string, echo exactly)
- query (a realistic user phrasing)
- gold_keywords (pipe-separated keywords, 2-6 terms)
- relevance_grade (integer 2 or 3)
- notes (short note)

Rules:
- Keep same intent as question.
- query should be natural and concise.
- gold_keywords should be specific terms likely present in evidence chunk.
- Use relevance_grade 3 for direct evidence, 2 for slightly broader wording.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def extract_json_array(text: str):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output does not contain a JSON array.")
    return json.loads(text[start : end + 1])


def fallback_rows_for_batch(batch: List[Dict[str, str]], source_filename: str) -> List[Dict[str, str]]:
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


def generate_first_pass_rows(
    parsed_questions: List[Dict[str, str]], model_name: str, source_filename: str
) -> List[Dict[str, str]]:
    llm = LLMService(model_name=model_name).llm
    rows = []

    batch_size = 10
    for i in range(0, len(parsed_questions), batch_size):
        batch = parsed_questions[i : i + batch_size]
        items = None
        for attempt in range(2):
            prompt = make_prompt(batch)
            if attempt == 1:
                prompt += "\nIMPORTANT: Return valid JSON array only. No prose."
            raw = llm.invoke(prompt)
            text = raw.content if hasattr(raw, "content") else str(raw)
            try:
                items = extract_json_array(text)
                break
            except Exception:
                items = None
        if items is None:
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
            row = {
                "query_id": query_id,
                "query": str(item.get("query", q["question"])).strip() or q["question"],
                "gold_source": source_filename,
                "gold_section_regex": rf"Q{q_no}\.",
                "gold_keywords": str(item.get("gold_keywords", "")).strip() or q["question"].lower(),
                "relevance_grade": str(item.get("relevance_grade", 3)),
                "notes": str(item.get("notes", f"Auto-generated from FAQ Q{q_no}")).strip(),
            }
            rows.append(row)

    # deterministic fallback if model skipped any questions
    existing_ids = {r["query_id"] for r in rows}
    for q in parsed_questions:
        qid = f"FAQ-{q['item_id']}"
        if qid in existing_ids:
            continue
        rows.append(
            {
                "query_id": qid,
                "query": q["question"],
                "gold_source": source_filename,
                "gold_section_regex": rf"Q{q['q_no']}\.",
                "gold_keywords": "|".join(
                    [w.lower() for w in re.findall(r"[A-Za-z]{4,}", q["question"])[:4]]
                ),
                "relevance_grade": "3",
                "notes": "Fallback row (LLM parse failed for this question)",
            }
        )

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
        description="Auto-generate first-pass qrels from FAQ.docx using an Ollama model."
    )
    parser.add_argument(
        "--faq-path",
        default="data/docs/Text_documents/FAQ.docx",
        help="Path to FAQ.docx",
    )
    parser.add_argument(
        "--output-path",
        default="data/benchmark_results/hr_qrels_from_faq_first_pass.csv",
        help="Output CSV path for generated qrels",
    )
    parser.add_argument(
        "--model-name",
        default="gemma2:2b",
        help="Ollama model for first-pass qrels generation",
    )
    args = parser.parse_args()

    faq_path = Path(args.faq_path)
    if not faq_path.exists():
        raise FileNotFoundError(f"FAQ not found at {faq_path}")

    parsed_questions = parse_faq_blocks(faq_path)
    if not parsed_questions:
        raise ValueError("No FAQ questions found in document.")

    rows = generate_first_pass_rows(
        parsed_questions=parsed_questions,
        model_name=args.model_name,
        source_filename=faq_path.name,
    )
    write_csv(rows, Path(args.output_path))
    print(f"Generated {len(rows)} qrel rows at: {args.output_path}")
    print("Review/edit before using for benchmark.")


if __name__ == "__main__":
    main()
