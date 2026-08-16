"""
Embedding evaluation using standard metrics from the MTEB / BEIR benchmarks,
instead of ad hoc heuristics.

Two evaluation families, matching how MTEB (Massive Text Embedding Benchmark)
and BEIR evaluate embedding models:

1. Retrieval (BEIR-style). Each paper's title is a short natural-language query,
   its own abstract is the single known-correct document. We report the standard
   IR metrics used on retrieval leaderboards: MRR, Recall@1, Recall@10, and
   nDCG@10 (the primary headline metric on MTEB/BEIR retrieval leaderboards).
   With a single relevant document per query, nDCG@10 reduces to
   1/log2(rank+1) if rank<=10 else 0, averaged over queries.

2. Clustering (MTEB Clustering-task style). We run k-means on the embeddings
   (k = number of target math subfields) and compare the resulting clusters to
   the papers' actual arXiv category using V-measure and Normalized Mutual
   Information (NMI) -- the standard metrics MTEB uses for its Clustering task.
   This replaces our earlier home-grown "category_overlap@10" heuristic with a
   recognized methodology.

A TF-IDF (keyword search) baseline is run through the same two evaluations, as
a sanity check on whether semantic embeddings actually beat keyword search.

Usage:
    python3 evaluate_embeddings.py
"""

import json
import math
import random
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import v_measure_score, normalized_mutual_info_score

DATA_PATH = "data/arxiv_math_subset.jsonl"
SAMPLE_SIZE = 5000
RANDOM_SEED = 42

TARGET_CATEGORIES = ["math.PR", "math.ST", "math.CO", "math.OC"]

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


def primary_category(record):
    """Single ground-truth label per paper, for clustering evaluation.
    Papers can carry several categories; we take the first of our four
    target subfields that appears."""
    cats = (record.get("categories") or "").split()
    for c in cats:
        if c in TARGET_CATEGORIES:
            return c
    return "other"


def retrieval_metrics(query_sims):
    """query_sims[i] = similarity of query i (title i) to every candidate
    document (abstract j). The correct document for query i is document i."""
    n = query_sims.shape[0]
    ranks = np.empty(n, dtype=int)
    for i in range(n):
        order = np.argsort(-query_sims[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1  # 1-indexed
    mrr = float(np.mean(1.0 / ranks))
    recall_at_1 = float(np.mean(ranks <= 1))
    recall_at_10 = float(np.mean(ranks <= 10))
    ndcg_at_10 = float(np.mean([1.0 / math.log2(r + 1) if r <= 10 else 0.0 for r in ranks]))
    return mrr, recall_at_1, recall_at_10, ndcg_at_10


def clustering_scores(embeddings, records):
    """MTEB Clustering-task style: k-means vs. true category labels."""
    true_labels = [primary_category(r) for r in records]
    n_clusters = len(set(true_labels))
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    pred_labels = km.fit_predict(embeddings)
    v = v_measure_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return v, nmi


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

    mrr, r1, r10, ndcg10 = retrieval_metrics(title_emb @ abstract_emb.T)
    v, nmi = clustering_scores(abstract_emb, records)

    print(f"dim={dim}  encode_time={elapsed:.1f}s")
    print(f"retrieval   MRR: {mrr:.3f}  Recall@1: {r1:.3f}  Recall@10: {r10:.3f}  nDCG@10: {ndcg10:.3f}")
    print(f"clustering  V-measure: {v:.3f}  NMI: {nmi:.3f}")

    return {
        "model": model_name, "dim": dim, "time_s": elapsed,
        "mrr": mrr, "recall@1": r1, "recall@10": r10, "ndcg@10": ndcg10,
        "v_measure": v, "nmi": nmi,
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
    dim = abstract_vecs.shape[1]

    mrr, r1, r10, ndcg10 = retrieval_metrics((title_vecs @ abstract_vecs.T).toarray())
    v, nmi = clustering_scores(abstract_vecs.toarray(), records)

    print(f"fit_transform_time={elapsed:.1f}s")
    print(f"retrieval   MRR: {mrr:.3f}  Recall@1: {r1:.3f}  Recall@10: {r10:.3f}  nDCG@10: {ndcg10:.3f}")
    print(f"clustering  V-measure: {v:.3f}  NMI: {nmi:.3f}")

    return {
        "model": "TF-IDF (keyword baseline)", "dim": dim, "time_s": elapsed,
        "mrr": mrr, "recall@1": r1, "recall@10": r10, "ndcg@10": ndcg10,
        "v_measure": v, "nmi": nmi,
    }


def main():
    records = load_sample()
    print(f"Loaded {len(records)} sample papers for comparison")

    results = [evaluate_sentence_transformer(m, records) for m in MODELS]
    results.append(evaluate_tfidf_baseline(records))

    print("\n\n=== Summary ===")
    print(f"{'model':38s} {'dim':>6s} {'MRR':>7s} {'R@1':>7s} {'R@10':>7s} {'nDCG@10':>8s} {'V-meas':>7s} {'NMI':>7s}")
    for r in results:
        print(f"{r['model']:38s} {r['dim']:6d} {r['mrr']:7.3f} {r['recall@1']:7.3f} "
              f"{r['recall@10']:7.3f} {r['ndcg@10']:8.3f} {r['v_measure']:7.3f} {r['nmi']:7.3f}")


if __name__ == "__main__":
    main()
