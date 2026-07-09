import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = [
    "query_id",
    "query",
    "gold_source",
    "gold_section_regex",
    "gold_keywords",
    "relevance_grade",
    "notes",
]

OPTIONAL_COLUMNS = ["gold_faq_id"]


def validate_qrels(csv_path: Path) -> int:
    errors = []
    warnings = []

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        return 1

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing_cols:
            errors.append(
                f"Missing required columns: {missing_cols}. "
                f"Found columns: {columns}"
            )
            print_report(errors, warnings)
            return 1

        seen_rows = set()
        query_by_id = {}
        rows_by_query_id = defaultdict(int)

        for row_num, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            query = (row.get("query") or "").strip()
            gold_source = (row.get("gold_source") or "").strip()
            gold_section_regex = (row.get("gold_section_regex") or "").strip()
            gold_keywords = (row.get("gold_keywords") or "").strip()
            gold_faq_id = (row.get("gold_faq_id") or "").strip()
            relevance_grade = (row.get("relevance_grade") or "").strip()

            if not query_id:
                errors.append(f"Row {row_num}: empty query_id")
            if not query:
                errors.append(f"Row {row_num}: empty query")
            if not gold_source:
                errors.append(f"Row {row_num}: empty gold_source")
            if not gold_keywords:
                errors.append(f"Row {row_num}: empty gold_keywords")
            if not relevance_grade:
                errors.append(f"Row {row_num}: empty relevance_grade")

            if query_id and query:
                if query_id in query_by_id and query_by_id[query_id] != query:
                    errors.append(
                        f"Row {row_num}: query_id '{query_id}' maps to multiple query texts."
                    )
                query_by_id[query_id] = query
                rows_by_query_id[query_id] += 1

            if relevance_grade:
                try:
                    grade = int(relevance_grade)
                    if grade < 0 or grade > 3:
                        errors.append(
                            f"Row {row_num}: relevance_grade '{relevance_grade}' must be in [0, 3]."
                        )
                except ValueError:
                    errors.append(
                        f"Row {row_num}: relevance_grade '{relevance_grade}' is not an integer."
                    )

            if gold_faq_id and "faq.docx" not in gold_source.lower():
                warnings.append(
                    f"Row {row_num}: gold_faq_id set but gold_source is not FAQ.docx."
                )

            if not gold_faq_id and not gold_section_regex:
                warnings.append(
                    f"Row {row_num}: neither gold_faq_id nor gold_section_regex is set."
                )

            if gold_section_regex:
                try:
                    re.compile(gold_section_regex)
                except re.error as ex:
                    if gold_faq_id:
                        warnings.append(
                            f"Row {row_num}: invalid gold_section_regex "
                            f"'{gold_section_regex}' (ignored when gold_faq_id is set): {ex}"
                        )
                    else:
                        errors.append(
                            f"Row {row_num}: invalid gold_section_regex "
                            f"'{gold_section_regex}': {ex}"
                        )

            keyword_parts = [k.strip() for k in gold_keywords.split("|") if k.strip()]
            if not keyword_parts:
                errors.append(
                    f"Row {row_num}: gold_keywords has no usable keywords (expected pipe-separated tokens)."
                )

            key = (
                query_id.lower(),
                query.lower(),
                gold_source.lower(),
                gold_section_regex,
                "|".join(keyword_parts).lower(),
                relevance_grade,
            )
            if key in seen_rows:
                warnings.append(f"Row {row_num}: duplicate qrel entry.")
            seen_rows.add(key)

        for qid, count in rows_by_query_id.items():
            if count < 1:
                warnings.append(f"query_id '{qid}' has zero qrels.")

    print_report(errors, warnings)
    return 1 if errors else 0


def print_report(errors, warnings):
    print("\nQrels Validation Report")
    print("=" * 30)
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors:
            print(f"- {e}")
    else:
        print("Errors: 0")

    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\nWarnings: 0")

    print("\nStatus:", "FAILED" if errors else "PASSED")


def main():
    parser = argparse.ArgumentParser(
        description="Strict validator for qrels-style retrieval benchmark CSV."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to qrels CSV (e.g. data/benchmark_results/hr_retrieval_golden_template.csv)",
    )
    args = parser.parse_args()

    exit_code = validate_qrels(Path(args.dataset_path))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
