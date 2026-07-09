"""
Patch FAQ qrels rows with gold_faq_id and section-specific gold_section_regex.

Maps FAQ-001..FAQ-043 (and paraphrases FAQ-001-p1, etc.) to unique chunk IDs
from the current DocumentService FAQ chunking.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.core.config import get_settings
from app.services.document_service import DocumentService

FAQ_BASE_ID = re.compile(r"^(FAQ-\d{3})")

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


def _build_section_regex(section: str, faq_question: str) -> str:
    snippet_match = re.search(r"Q\d+\.\s*(.+)", faq_question)
    snippet = re.sub(
        r"\s+", " ", (snippet_match.group(1) if snippet_match else "")
    ).strip()
    words = snippet.split()[:6]
    safe_snippet = re.escape(" ".join(words)).rstrip("\\")
    safe_section = re.escape(section).rstrip("\\")
    if safe_section and safe_snippet:
        return f"{safe_section}.*{safe_snippet}"
    return safe_snippet or safe_section


def build_faq_lookup() -> Dict[str, Dict[str, str]]:
    settings = get_settings()
    ds = DocumentService()
    docs = ds.load_documents(str(settings.DOCS_DIR))
    chunks = [
        c
        for c in ds.split_documents(docs)
        if "faq" in c.metadata.get("source", "").lower()
    ]

    lookup: Dict[str, Dict[str, str]] = {}
    for i, chunk in enumerate(chunks, start=1):
        base_id = f"FAQ-{i:03d}"
        section = str(chunk.metadata.get("section", ""))
        faq_id = str(chunk.metadata.get("faq_id", ""))
        faq_question = str(chunk.metadata.get("faq_question", ""))

        lookup[base_id] = {
            "gold_faq_id": faq_id,
            "gold_section_regex": _build_section_regex(section, faq_question),
        }
    return lookup


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def patch_rows(rows: List[Dict[str, str]], lookup: Dict[str, Dict[str, str]]) -> int:
    patched = 0
    for row in rows:
        if "faq.docx" not in str(row.get("gold_source", "")).lower():
            continue
        match = FAQ_BASE_ID.match(str(row.get("query_id", "")).strip())
        if not match:
            continue
        info = lookup.get(match.group(1))
        if not info:
            continue
        row["gold_faq_id"] = info["gold_faq_id"]
        row["gold_section_regex"] = info["gold_section_regex"]
        patched += 1
    return patched


def main():
    parser = argparse.ArgumentParser(
        description="Patch FAQ qrels with gold_faq_id and unique section regex."
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

    lookup = build_faq_lookup()
    rows = load_rows(input_path)
    count = patch_rows(rows, lookup)
    write_rows(output_path, rows)

    print(f"Patched {count} FAQ rows in {output_path}")
    print(f"FAQ lookup entries: {len(lookup)}")
    print("Sample mappings:")
    for base in ("FAQ-001", "FAQ-013", "FAQ-027", "FAQ-042"):
        if base in lookup:
            print(f"  {base} -> {lookup[base]['gold_faq_id']}")


if __name__ == "__main__":
    main()
