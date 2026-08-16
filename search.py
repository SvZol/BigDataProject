"""
Quick end-to-end test of semantic search: embeds a natural-language query with
the same model used to build the index (all-mpnet-base-v2), runs a kNN search
against Elasticsearch, and prints the top matches.

Usage:
    python3 search.py "concentration inequalities for random matrices"
    python3 search.py "some paper title and abstract here" --since 2025-01-01 --k 5
"""

import argparse

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_NAME = "arxiv_math"
ES_HOST = "http://localhost:9200"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Natural language search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only papers submitted on/after this date, e.g. 2025-01-01 (for 'recent only' searches)",
    )
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    es = Elasticsearch(ES_HOST)

    query_vector = model.encode(args.query, normalize_embeddings=True).tolist()

    knn = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": args.k,
        "num_candidates": max(100, args.k * 20),
    }
    if args.since:
        knn["filter"] = {"range": {"submitted": {"gte": args.since}}}

    resp = es.search(
        index=INDEX_NAME, knn=knn, size=args.k,
        source=["title", "authors", "categories", "submitted"],
    )

    print(f"\nQuery: {args.query}")
    if args.since:
        print(f"(filtered to papers submitted since {args.since})")
    print()

    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        print(f"[{hit['_score']:.3f}] {src['title']}")
        print(f"    categories: {', '.join(src.get('categories', []))}   submitted: {src.get('submitted')}")
        print(f"    authors: {src.get('authors')}\n")


if __name__ == "__main__":
    main()
