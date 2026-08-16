"""
Embeds every paper in the filtered arXiv math subset with the chosen model
(all-mpnet-base-v2, picked via evaluate_embeddings.py) and bulk-loads the
result into the Elasticsearch index `arxiv_math`.

Text embedded per paper: "{title}. {abstract}" (same convention used during
model evaluation).

Usage (from project root):
    python3 build_index.py
"""

import json
import time

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/arxiv_math_subset.jsonl"
INDEX_NAME = "arxiv_math"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
ES_HOST = "http://localhost:9200"
BATCH_SIZE = 250


def read_batches(path, batch_size):
    batch = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def to_action(record, embedding):
    return {
        "_index": INDEX_NAME,
        "_id": record["id"],
        "_source": {
            "id": record.get("id"),
            "title": record.get("title"),
            "authors": record.get("authors"),
            "categories": (record.get("categories") or "").split(),
            "abstract": record.get("abstract"),
            "submitted": record.get("submitted"),
            "update_date": record.get("update_date"),
            "embedding": embedding.tolist(),
        },
    }


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise RuntimeError(f"Could not reach Elasticsearch at {ES_HOST}")

    start = time.time()
    total = 0

    for batch in read_batches(DATA_PATH, BATCH_SIZE):
        texts = [f"{r['title']}. {r['abstract']}" for r in batch]
        embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)

        actions = [to_action(r, e) for r, e in zip(batch, embeddings)]
        helpers.bulk(es, actions)

        total += len(batch)
        elapsed = time.time() - start
        rate = total / elapsed if elapsed > 0 else 0
        print(f"indexed {total:,} docs  ({elapsed:.0f}s elapsed, {rate:.1f} docs/s)")

    print(f"\nDone. Indexed {total:,} documents into '{INDEX_NAME}' in {time.time() - start:.0f}s.")


if __name__ == "__main__":
    main()
