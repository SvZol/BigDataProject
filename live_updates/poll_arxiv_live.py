"""
Polls the REAL arXiv API for papers submitted since the last check, in our
four target categories (math.PR, math.ST, math.CO, math.OC), and publishes
any genuinely new ones to a separate Kafka topic: arxiv-papers-live.

This is what makes "regularly updated" actually true, unlike
kafka_producer.py (a one-time replay of the static historical dataset this
project is built on). Every run of this script talks to the live arXiv API
(export.arxiv.org) and only publishes papers that weren't already seen on a
previous run -- state is tracked per category in data/live_poll_state.json.

We use a SEPARATE topic (arxiv-papers-live) rather than reusing
arxiv-papers, so the one-time historical replay and the ongoing live feed
stay clearly distinguishable both in the code and if anyone inspects the
Kafka topics directly.

First run: instead of bootstrapping to "now" (which would mean waiting for
organically new papers before anything shows up), we bootstrap the cutoff
per category to the newest "submitted" date already present in our existing
filtered dataset (data/arxiv_math_subset.jsonl). That dataset was collected
at some fixed point in the past, so this immediately finds everything
genuinely published in that category since then -- real papers, not a
simulated window. Subsequent runs use the state file and only look for
what's new since the previous check. --since-hours is still available as a
manual override (e.g. to force an even wider check on demand).

Usage:
    python3 live_updates/poll_arxiv_live.py
    python3 live_updates/poll_arxiv_live.py --since-hours 48
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from kafka import KafkaProducer

TARGET_CATEGORIES = ["math.PR", "math.ST", "math.CO", "math.OC"]
STATE_PATH = "data/live_poll_state.json"
DATASET_PATH = "data/arxiv_math_subset.jsonl"
API_URL = "http://export.arxiv.org/api/query"
MAX_RESULTS = 50           # per category, per run -- plenty for a "since last check" delta
SLEEP_BETWEEN_CATEGORIES = 3.0  # arXiv API etiquette: don't hammer it
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "arxiv-papers-live"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def parse_iso(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def bootstrap_state_from_dataset():
    """Per-category cutoff = the newest 'submitted' date already present in
    our existing filtered dataset. Used only for categories missing from the
    state file (normally just on the very first run) -- lets the first real
    check find genuine new papers since the dataset was collected, instead
    of an arbitrary window from 'now'."""
    newest = {}
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                submitted = record.get("submitted")
                if not submitted:
                    continue
                for cat in (record.get("categories") or "").split():
                    if cat in TARGET_CATEGORIES and (cat not in newest or submitted > newest[cat]):
                        newest[cat] = submitted
    except FileNotFoundError:
        return {}
    return {cat: f"{date}T00:00:00+00:00" for cat, date in newest.items()}


def fetch_category(category, retries=3):
    params = f"search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS}"
    url = f"{API_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "student-thesis-search-project/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            wait = 5 * (attempt + 1)
            print(f"  ({category}) network error ({e}), retrying in {wait}s...")
            time.sleep(wait)
    print(f"  ({category}) failed after {retries} attempts, skipping this category")
    return None


def parse_entries(xml_bytes):
    root = ET.fromstring(xml_bytes)
    entries = []
    for entry in root.findall("atom:entry", NS):
        id_text = entry.findtext("atom:id", default="", namespaces=NS)
        arxiv_id = re.sub(r"v\d+$", "", id_text.rsplit("/", 1)[-1])
        if not arxiv_id:
            continue

        title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip().replace("\n", " ")
        published_text = entry.findtext("atom:published", default="", namespaces=NS)
        updated_text = entry.findtext("atom:updated", default="", namespaces=NS)
        authors = [a.findtext("atom:name", default="", namespaces=NS) for a in entry.findall("atom:author", NS)]
        categories = [c.get("term") for c in entry.findall("atom:category", NS) if c.get("term")]

        if not published_text:
            continue
        published = parse_iso(published_text)

        entries.append({
            "id": arxiv_id,
            "title": title,
            "authors": ", ".join(a for a in authors if a),
            "categories": " ".join(categories),
            "abstract": abstract,
            "submitted": published.strftime("%Y-%m-%d"),
            "update_date": parse_iso(updated_text).strftime("%Y-%m-%d") if updated_text else None,
            "_published_dt": published,
        })
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--since-hours", type=float, default=None,
        help="Force a look-back window of this many hours instead of using the saved "
             "state / dataset cutoff. Overrides both. Papers found are still real.",
    )
    args = parser.parse_args()

    state = load_state()
    now = datetime.now(timezone.utc)
    dataset_bootstrap = bootstrap_state_from_dataset()

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    all_new = {}  # id -> record, deduped across categories (a paper can carry several of our categories)
    for category in TARGET_CATEGORIES:
        print(f"Checking {category}...")
        xml_bytes = fetch_category(category)
        if xml_bytes is None:
            continue
        entries = parse_entries(xml_bytes)

        if entries:
            newest = max(entries, key=lambda e: e["_published_dt"])
            print(f"  fetched {len(entries)} entries, newest: {newest['submitted']} ({newest['id']})")
        else:
            print(f"  fetched 0 entries -- check search_query/parsing if this looks wrong")

        if args.since_hours is not None:
            # explicit flag always wins, even if state already exists -- lets you force
            # a wider look-back on demand (e.g. right before a demo) without deleting state
            cutoff = now - timedelta(hours=args.since_hours)
            print(f"  cutoff: {cutoff.isoformat()[:10]} (--since-hours {args.since_hours} override)")
        elif category in state:
            cutoff = parse_iso(state[category])
            print(f"  cutoff: {cutoff.isoformat()[:10]} (from saved state, data/live_poll_state.json)")
        elif category in dataset_bootstrap:
            cutoff = parse_iso(dataset_bootstrap[category])
            print(f"  cutoff: {cutoff.isoformat()[:10]} (no saved state -- newest date already in our dataset)")
        else:
            cutoff = None  # nothing to compare against -- publish nothing, just record a baseline
            print(f"  cutoff: none (no state, no dataset match) -- publishing nothing, recording baseline only")

        newest_seen = state.get(category)
        for e in entries:
            if newest_seen is None or e["_published_dt"] > parse_iso(newest_seen):
                newest_seen = e["submitted"] + "T00:00:00+00:00"  # coarse but monotonic enough
            if cutoff is not None and e["_published_dt"] > cutoff:
                all_new[e["id"]] = e

        if entries:
            state[category] = max(entries, key=lambda e: e["_published_dt"])["_published_dt"].isoformat()

        time.sleep(SLEEP_BETWEEN_CATEGORIES)

    for record in all_new.values():
        record.pop("_published_dt", None)
        producer.send(TOPIC, value=record)

    producer.flush()
    producer.close()
    save_state(state)

    print(f"\nDone. Found {len(all_new)} genuinely new paper(s) across {len(state)} categories, "
          f"published to Kafka topic '{TOPIC}'.")


if __name__ == "__main__":
    main()
