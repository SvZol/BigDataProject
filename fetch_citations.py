"""
Fetches real citation counts for every paper in the filtered arXiv math subset,
using the Semantic Scholar Graph API batch endpoint (no API key required).
Writes a lookup file data/citations.jsonl with {id, citation_count}, which we
later join with the main dataset for KPI/insight computation (e.g. relating
keyword/topic trends to citation impact).

Resumable: if the script is interrupted (network error, Ctrl+C), re-running it
skips ids that are already in the output file and continues from there.

Usage:
    python3 fetch_citations.py
"""

import json
import time
import urllib.error
import urllib.request

DATA_PATH = "data/arxiv_math_subset.jsonl"
OUTPUT_PATH = "data/citations.jsonl"
BATCH_SIZE = 500
API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=citationCount"
SLEEP_BETWEEN_BATCHES = 3.0  # seconds -- be polite to the shared unauthenticated pool
REQUEST_TIMEOUT = 60


def load_ids():
    ids = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["id"])
    return ids


def load_already_fetched():
    fetched = set()
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fetched.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    # last line may be incomplete if a previous run crashed mid-write
                    continue
    except FileNotFoundError:
        pass
    return fetched


def fetch_batch(ids, retries=5):
    payload = json.dumps({"ids": [f"ARXIV:{i}" for i in ids]}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"  rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  HTTP error {e.code} on this batch, skipping ({e.reason})")
            return [None] * len(ids)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            wait = 5 * (attempt + 1)
            print(f"  network error ({e}), retrying in {wait}s...")
            time.sleep(wait)
            continue
    print("  too many retries, skipping this batch")
    return [None] * len(ids)


def main():
    ids = load_ids()
    already = load_already_fetched()
    remaining = [i for i in ids if i not in already]
    print(f"Loaded {len(ids):,} paper ids, {len(already):,} already fetched, {len(remaining):,} remaining")

    if not remaining:
        print("Nothing left to fetch.")
        return

    found = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[start:start + BATCH_SIZE]
            results = fetch_batch(batch)

            for paper_id, result in zip(batch, results):
                citation_count = result.get("citationCount") if result else None
                if citation_count is not None:
                    found += 1
                out.write(json.dumps({"id": paper_id, "citation_count": citation_count}) + "\n")
            out.flush()

            done = start + len(batch)
            print(f"fetched {done:,}/{len(remaining):,} remaining papers (matched so far: {found:,})")
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\nDone. Wrote/updated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
