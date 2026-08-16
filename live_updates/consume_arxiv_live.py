"""
Drains the Kafka topic arxiv-papers-live (real new papers found by
poll_arxiv_live.py), embeds each one with the same model used for the main
index (all-mpnet-base-v2), and upserts it directly into the existing
Elasticsearch index arxiv_math -- so it's searchable / usable by RAG
immediately, without rebuilding the whole 57k-document index.

Also appends each new paper to data/arxiv_math_live_additions.jsonl -- kept
as a separate file from the original static dataset so it's always clear,
by provenance, which papers came from the one-time historical load and
which arrived through the live pipeline.

Two dedup safeguards, both needed in practice:
  1. group_id on the consumer, so Kafka tracks what this script has already
     read -- without it, every run rereads the ENTIRE topic from the start
     (no persisted offset), reprocessing everything each time.
  2. An in-memory id dedup during the run, because Kafka producers are
     "at least once" by default -- a retried send can land the same message
     in the topic twice, which we saw happen (every paper appeared exactly
     twice in one run). Elasticsearch upserts by id so that side is
     naturally safe, but the append-only local file is not, so we dedup
     there explicitly (against ids already in the file too, in case an
     older run left duplicates before this fix).

Usage:
    python3 live_updates/consume_arxiv_live.py
"""

import json

from elasticsearch import Elasticsearch
from kafka import KafkaConsumer
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_NAME = "arxiv_math"
ES_HOST = "http://localhost:9200"
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "arxiv-papers-live"
GROUP_ID = "arxiv_live_indexer"
OUTPUT_PATH = "data/arxiv_math_live_additions.jsonl"
IDLE_TIMEOUT_MS = 10_000  # one-shot drain per run, same convention as kafka_consumer_to_file.py


def load_already_added():
    seen = set()
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return seen


def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=IDLE_TIMEOUT_MS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise RuntimeError(f"Could not reach Elasticsearch at {ES_HOST}")

    already_added = load_already_added()
    seen_this_run = set()
    added = 0
    skipped_dupes = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for message in consumer:
            record = message.value
            paper_id = record["id"]

            if paper_id in already_added or paper_id in seen_this_run:
                skipped_dupes += 1
                continue
            seen_this_run.add(paper_id)

            text = f"{record['title']}. {record['abstract']}"
            embedding = model.encode(text, normalize_embeddings=True)

            es.index(
                index=INDEX_NAME,
                id=paper_id,
                document={
                    "id": record.get("id"),
                    "title": record.get("title"),
                    "authors": record.get("authors"),
                    "categories": (record.get("categories") or "").split(),
                    "abstract": record.get("abstract"),
                    "submitted": record.get("submitted"),
                    "update_date": record.get("update_date"),
                    "embedding": embedding.tolist(),
                },
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            added += 1
            print(f"  indexed {paper_id}: {record['title'][:70]}")

    consumer.close()
    es.indices.refresh(index=INDEX_NAME)  # make new docs visible to search immediately, not just eventually
    print(f"\nDone. Added {added} new paper(s) to '{INDEX_NAME}' -- searchable right now "
          f"({skipped_dupes} duplicate message(s) skipped).")


if __name__ == "__main__":
    main()
