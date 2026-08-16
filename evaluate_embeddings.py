"""
Extended embedding evaluation. Builds on the first comparison (category_overlap@10)
and adds two more rigorous checks:

1. Title -> Abstract retrieval (self-supervised, no manual labeling needed).
   Each paper's title is used as a short natural-language query, and its own
   abstract is the known-correct document. We measure how well each model
   retrieves the correct abstract among all abstracts in the sample:
   MRR, Recall@1, Recall@10. This mirrors how the real system will be used
   (short query -> find the right document) much more closely than clustering
   by arXiv category, which is a coarse, author-assigned label.

2. TF-IDF baseline (plain keyword search), run through the exact same
   title->abstract retrieval task. This directly answers the brief's question:
   does semantic (embedding) search actually beat keyword search on our data?

category_overlap@10 is still reported (now computed on abstract embeddings)
as a secondary, cheap sanity signal.

Usage:
    python3 evaluate_embeddings.py
"""

import json
import random
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data/arxiv_math_subset.jsonl"
SAMPLE_SIZE = 5000
TOP_K = 10
RANDOM_SEED = 42

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/allenai-specter",
]


def load_sample():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    random.seed(RANDOM_SEED)
    return random.sample(records, min(SAMPLE_SIZE, len(records)))


def category_set(record):
    return set((record.get("categories") or "").split())


def category_overlap_at_k(sims, records, k=TOP_K):
    scores = []
    for i, record in enumerate(records):
        top_idx = np.argsort(-sims[i])[:k]
        own_cats = category_set(record)
        overlap = sum(1 for j in top_idx if own_cats & category_set(records[j]))
        scores.append(overlap / k)
    return float(np.mean(scores))


def retrieval_metrics(query_sims):
    """query_sims[i] = similarity of query i to every candidate document.
    The correct document for query i is document i (title i matches abstract i)."""
    n = query_sims.shape[0]
    ranks = np.empty(n, dtype=int)
    for i in range(n):
        order = np.argsort(-query_sims[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1  # 1-indexed rank
    mrr = float(np.mean(1.0 / ranks))
    recall_at_1 = float(np.mean(ranks <= 1))
    recall_at_10 = float(np.mean(ranks <= 10))
    return mrr, recall_at_1, recall_at_10


def evaluate_sentence_transformer(model_name, records):
    print(f"\n=== {model_name} ===")
    model = SentenceTransformer(model_name)
    titles = [r["title"] for r in records]
    abstracts = [r["abstract"] for r in records]

    start = time.time()
    title_emb = model.encode(titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    abstract_emb = model.encode(abstracts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    elapsed = time.time() - start

    dim = abstract_emb.shape[1]

    # category coherence, using abstract embeddings as the representation
    abs_sims = abstract_emb @ abstract_emb.T
    np.fill_diagonal(abs_sims, -1.0)
    overlap = category_overlap_at_k(abs_sims, records)

    # title -> abstract retrieval
    query_sims = title_emb @ abstract_emb.T
    mrr, r1, r10 = retrieval_metrics(query_sims)

    print(f"dim={dim}  encode_time={elapsed:.1f}s")
    print(f"category_overlap@{TOP_K}: {overlap:.3f}")
    print(f"title->abstract retrieval  MRR: {mrr:.3f}  Recall@1: {r1:.3f}  Recall@10: {r10:.3f}")

    return {
        "model": model_name, "dim": dim, "time_s": elapsed,
        "overlap": overlap, "mrr": mrr, "recall@1": r1, "recall@10": r10,
    }


def evaluate_tfidf_baseline(records):
    print("\n=== TF-IDF baseline (keyword search) ===")
    titles = [r["title"] for r in records]
    abstracts = [r["abstract"] for r in records]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    start = time.time()
    abstract_vecs = vectorizer.fit_transform(abstracts)
    title_vecs = vectorizer.transform(titles)
    elapsed = time.time() - start

    query_sims = (title_vecs @ abstract_vecs.T).toarray()
    mrr, r1, r10 = retrieval_metrics(query_sims)

    abs_sims = (abstract_vecs @ abstract_vecs.T).toarray()
    np.fill_diagonal(abs_sims, -1.0)
    overlap = category_overlap_at_k(abs_sims, records)

    print(f"fit_transform_time={elapsed:.1f}s")
    print(f"category_overlap@{TOP_K}: {overlap:.3f}")
    print(f"title->abstract retrieval  MRR: {mrr:.3f}  Recall@1: {r1:.3f}  Recall@10: {r10:.3f}")

    return {
        "model": "TF-IDF (keyword baseline)", "dim": abstract_vecs.shape[1], "time_s": elapsed,
        "overlap": overlap, "mrr": mrr, "recall@1": r1, "recall@10": r10,
    }


def main():
    records = load_sample()
    print(f"Loaded {len(records)} sample papers for comparison")

    results = [evaluate_sentence_transformer(m, records) for m in MODELS]
    results.append(evaluate_tfidf_baseline(records))

    print("\n\n=== Summary ===")
    print(f"{'model':38s} {'dim':>6s} {'overlap@10':>11s} {'MRR':>7s} {'R@1':>7s} {'R@10':>7s}")
    for r in results:
        print(f"{r['model']:38s} {r['dim']:6d} {r['overlap']:11.3f} {r['mrr']:7.3f} {r['recall@1']:7.3f} {r['recall@10']:7.3f}")


if __name__ == "__main__":
    main()
