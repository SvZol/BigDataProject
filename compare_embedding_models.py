"""
Compares candidate sentence-embedding models on a sample of the arXiv math subset.

For each model we report:
  - embedding dimension
  - encoding time (for a fixed-size sample)
  - category_overlap@10: an unsupervised quality proxy. For each paper, we find its
    10 nearest neighbors by cosine similarity and measure what fraction of them share
    at least one arXiv category with the paper itself. Meaningful embeddings should
    cluster same-topic papers together, so a higher score suggests embeddings that
    better capture mathematical topic structure -- without needing any hand-labeled
    relevance judgments.
  - a few example nearest-neighbor lists, for manual sanity-checking.

Usage:
    python3 compare_embedding_models.py
"""

import json
import random
import time

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/arxiv_math_subset.jsonl"
SAMPLE_SIZE = 5000
TOP_K = 10
RANDOM_SEED = 42
NUM_EXAMPLE_QUERIES = 5

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",     # small, fast, general-purpose
    "sentence-transformers/all-mpnet-base-v2",    # larger, stronger general-purpose
    "sentence-transformers/allenai-specter",      # trained specifically on scientific papers
]


def load_sample():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    random.seed(RANDOM_SEED)
    return random.sample(records, min(SAMPLE_SIZE, len(records)))


def category_set(record):
    return set((record.get("categories") or "").split())


def evaluate_model(model_name, records):
    print(f"\n=== {model_name} ===")
    model = SentenceTransformer(model_name)
    texts = [f"{r['title']}. {r['abstract']}" for r in records]

    start = time.time()
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    elapsed = time.time() - start

    dim = embeddings.shape[1]
    print(f"dim={dim}  encode_time_for_{len(records)}_docs={elapsed:.1f}s")

    # embeddings are normalized, so dot product == cosine similarity
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1.0)  # exclude self-match

    overlap_scores = []
    for i, record in enumerate(records):
        top_idx = np.argsort(-sims[i])[:TOP_K]
        own_cats = category_set(record)
        overlap = sum(1 for j in top_idx if own_cats & category_set(records[j]))
        overlap_scores.append(overlap / TOP_K)

    avg_overlap = float(np.mean(overlap_scores))
    print(f"category_overlap@{TOP_K} (avg over {len(records)} papers): {avg_overlap:.3f}")

    print("\nExample neighbors (for manual sanity-check):")
    for i in random.sample(range(len(records)), NUM_EXAMPLE_QUERIES):
        top_idx = np.argsort(-sims[i])[:3]
        print(f"\n  Query: {records[i]['title']}  [{records[i]['categories']}]")
        for j in top_idx:
            print(f"    -> {records[j]['title']}  [{records[j]['categories']}]  sim={sims[i][j]:.3f}")

    return {"model": model_name, "dim": dim, "encode_time_s": elapsed, "avg_overlap": avg_overlap}


def main():
    records = load_sample()
    print(f"Loaded {len(records)} sample papers for comparison")

    results = [evaluate_model(m, records) for m in MODELS]

    print("\n\n=== Summary ===")
    print(f"{'model':45s} {'dim':>5s} {'encode_s':>10s} {'overlap@10':>12s}")
    for r in results:
        print(f"{r['model']:45s} {r['dim']:5d} {r['encode_time_s']:10.1f} {r['avg_overlap']:12.3f}")


if __name__ == "__main__":
    main()
