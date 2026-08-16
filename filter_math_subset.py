"""
Filters the full arXiv metadata dump down to a subset:
categories math.PR, math.ST, math.CO, math.OC from the last 3 years.

Usage (from project root):
    python3 filter_math_subset.py
"""

import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

INPUT_PATH = "raw/arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH = "data/arxiv_math_subset.jsonl"

TARGET_CATEGORIES = {"math.PR", "math.ST", "math.CO", "math.OC"}
YEARS_BACK = 3


def get_submission_date(record):
    """Date of the first version of the paper (a more honest 'publication date'), falling back to update_date."""
    versions = record.get("versions") or []
    if versions:
        raw = versions[0].get("created")
        if raw:
            try:
                return parsedate_to_datetime(raw)
            except Exception:
                pass
    update_date = record.get("update_date")
    if update_date:
        try:
            return datetime.strptime(update_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def matches_category(record):
    cats = record.get("categories", "") or ""
    tokens = set(cats.split())
    return bool(tokens & TARGET_CATEGORIES)


def main():
    now = datetime.now(timezone.utc)
    cutoff = now.replace(year=now.year - YEARS_BACK)

    total = 0
    kept = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:

        for line in infile:
            total += 1
            if total % 200_000 == 0:
                print(f"processed {total:,} lines, kept {kept:,}")

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not matches_category(record):
                continue

            sub_date = get_submission_date(record)
            if sub_date is None or sub_date < cutoff:
                continue

            out = {
                "id": record.get("id"),
                "title": (record.get("title") or "").strip().replace("\n", " "),
                "authors": record.get("authors"),
                "categories": record.get("categories"),
                "abstract": (record.get("abstract") or "").strip().replace("\n", " "),
                "submitted": sub_date.strftime("%Y-%m-%d"),
                "update_date": record.get("update_date"),
            }
            outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. Total lines: {total:,}. Kept: {kept:,}. Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
