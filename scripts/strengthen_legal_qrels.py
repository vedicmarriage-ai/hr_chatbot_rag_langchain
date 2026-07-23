"""
Strengthen legal/refund qrels matching using live corpus chunk headers.

For non-FAQ rows:
- Prefers gold_section_regex built from legal_heading / section / clause markers
- Adds optional gold_legal_id when resolvable
- Keeps keywords that actually appear in the matched chunk text

Example:
  python scripts/strengthen_legal_qrels.py \\
    --input-path data/benchmark_results/hr_qrels_full_first_pass_for_review.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.core.config import get_settings
from app.services.document_service import DocumentService

FIELDNAMES = [
    "query_id",
    "query",
    "gold_source",
    "gold_faq_id",
    "gold_legal_id",
    "gold_section_regex",
    "gold_keywords",
    "relevance_grade",
    "notes",
]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_legal_chunks() -> List:
    settings = get_settings()
    ds = DocumentService()
    docs = ds.load_documents(str(settings.DOCS_DIR))
    chunks = ds.split_documents(docs)
    return [
        c
        for c in chunks
        if "faq" not in Path(c.metadata.get("source", "")).name.lower()
    ]


def _escape_safe(text: str, max_len: int = 70) -> str:
    clipped = text[:max_len].rstrip()
    # Avoid ending on a truncated escape sequence when validating regex
    while clipped and clipped[-1] in "\\ ":
        clipped = clipped[:-1]
    # Prefer cutting at a word/punctuation boundary
    if len(text) > max_len:
        cut = max(clipped.rfind(" "), clipped.rfind("."), clipped.rfind(":"))
        if cut >= 20:
            clipped = clipped[:cut]
    return re.escape(clipped).rstrip("\\")


def make_section_regex(chunk) -> str:
    heading = str(chunk.metadata.get("legal_heading") or "").strip()
    section = str(chunk.metadata.get("section") or "").strip()
    content = chunk.page_content

    clause = re.search(r"(\d+\.\d+\.)", content)
    if clause:
        # Short stable clause marker avoids trailing-escape truncation bugs
        return re.escape(clause.group(1))

    if heading and heading.lower() != "introduction":
        return _escape_safe(heading, 60)

    if section and section.lower() != "introduction":
        return _escape_safe(section, 60)

    first_line = content.strip().split("\n")[0][:60]
    return _escape_safe(first_line, 60) if first_line else "."


def keywords_from_chunk(chunk, existing: str) -> str:
    text = chunk.page_content.lower()
    kept = []
    for token in [t.strip() for t in existing.split("|") if t.strip()]:
        if token.lower() in text:
            kept.append(token)
    if len(kept) >= 2:
        return "|".join(kept)

    # Fall back to discriminative tokens from heading + body
    heading = str(chunk.metadata.get("legal_heading") or chunk.metadata.get("section") or "")
    words = re.findall(r"[A-Za-z]{4,}", f"{heading} {chunk.page_content[:300]}")
    uniq = []
    for w in words:
        wl = w.lower()
        if wl not in uniq and wl not in {"that", "this", "with", "from", "have", "will"}:
            uniq.append(wl)
        if len(uniq) >= 5:
            break
    return "|".join(uniq) or "policy|terms"


def score_chunk(query: str, keywords: str, chunk) -> float:
    text = chunk.page_content.lower()
    q_tokens = set(re.findall(r"[a-z0-9]{4,}", query.lower()))
    text_tokens = set(re.findall(r"[a-z0-9]{4,}", text))
    overlap = len(q_tokens & text_tokens) / max(1, len(q_tokens))
    kw_hits = 0
    kw_parts = [k.strip().lower() for k in keywords.split("|") if k.strip()]
    if kw_parts:
        kw_hits = sum(1 for k in kw_parts if k in text) / len(kw_parts)
    heading = str(chunk.metadata.get("legal_heading") or "").lower()
    heading_bonus = 0.15 if any(t in heading for t in q_tokens) else 0.0
    return overlap + 0.5 * kw_hits + heading_bonus


def best_chunk_for_row(row: Dict[str, str], chunks: List) -> Optional[object]:
    source = Path(str(row.get("gold_source", ""))).name.lower()
    candidates = [
        c
        for c in chunks
        if Path(c.metadata.get("source", "")).name.lower() == source
        or source in Path(c.metadata.get("source", "")).name.lower()
    ]
    if not candidates:
        # looser: any legal chunk containing current regex
        regex = row.get("gold_section_regex") or ""
        for c in chunks:
            try:
                if regex and re.search(regex, c.page_content, re.IGNORECASE):
                    candidates.append(c)
            except re.error:
                continue
    if not candidates:
        return None

    query = row.get("query", "")
    keywords = row.get("gold_keywords", "")
    ranked = sorted(
        candidates,
        key=lambda c: score_chunk(query, keywords, c),
        reverse=True,
    )
    return ranked[0]


def strengthen_rows(rows: List[Dict[str, str]], chunks: List) -> Tuple[List[Dict[str, str]], int]:
    updated = 0
    out: List[Dict[str, str]] = []
    for row in rows:
        source = str(row.get("gold_source", "")).lower()
        new_row = dict(row)
        new_row.setdefault("gold_faq_id", row.get("gold_faq_id", ""))
        new_row.setdefault("gold_legal_id", "")

        if "faq" in source:
            out.append(new_row)
            continue

        chunk = best_chunk_for_row(row, chunks)
        if not chunk:
            out.append(new_row)
            continue

        new_row["gold_section_regex"] = make_section_regex(chunk)
        new_row["gold_keywords"] = keywords_from_chunk(
            chunk, row.get("gold_keywords", "")
        )
        new_row["gold_legal_id"] = str(chunk.metadata.get("legal_id") or "")
        updated += 1
        out.append(new_row)
    return out, updated


def main():
    parser = argparse.ArgumentParser(
        description="Strengthen legal qrels regex/keywords from corpus chunks."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Input qrels CSV",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Output CSV (default: overwrite input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path

    chunks = build_legal_chunks()
    rows = load_rows(input_path)
    strengthened, count = strengthen_rows(rows, chunks)
    write_rows(output_path, strengthened)
    print(f"Strengthened {count} legal/refund rows in {output_path}")


if __name__ == "__main__":
    main()
