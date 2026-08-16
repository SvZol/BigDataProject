"""
RAG: answers a natural-language question about our arXiv math subset by
retrieving the most relevant abstracts (same semantic search as search.py)
and asking a local LLM (via Ollama) to answer using only that retrieved
context, citing which papers it actually used.

Usage:
    python3 search_ai/rag_answer.py "What are recent approaches to concentration inequalities for random matrices?"
    python3 search_ai/rag_answer.py "some question" --since 2025-01-01 --k 5
"""

import argparse

import ollama
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_NAME = "arxiv_math"
ES_HOST = "http://localhost:9200"
OLLAMA_MODEL = "llama3.1:8b"  # chosen over llama3.2 (3B) per the head-to-head comparison -- see project_brief.md
TOP_K = 5


def retrieve(question, embed_model, es, k, since=None):
    query_vector = embed_model.encode(question, normalize_embeddings=True).tolist()
    knn = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": max(100, k * 20),
    }
    if since:
        knn["filter"] = {"range": {"submitted": {"gte": since}}}

    resp = es.search(
        index=INDEX_NAME, knn=knn, size=k,
        source=["id", "title", "authors", "categories", "abstract", "submitted"],
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def build_prompt(question, papers):
    context_blocks = []
    for i, p in enumerate(papers, start=1):
        context_blocks.append(
            f"[{i}] Title: {p['title']}\n"
            f"Authors: {p['authors']}\n"
            f"Categories: {', '.join(p.get('categories', []))}\n"
            f"Submitted: {p.get('submitted')}\n"
            f"Abstract: {p['abstract']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""You are a research assistant answering questions about mathematics papers (probability, statistics, combinatorics, optimization).

Use ONLY the paper abstracts below to answer the question. Do not use any outside knowledge. If the abstracts do not contain enough information to answer, say so explicitly instead of guessing.

Synthesize across the papers rather than summarizing one at a time: compare and contrast their approaches where relevant, and draw on as many of the provided papers as are actually relevant to the question. Paraphrase in your own words instead of copying phrases directly from the abstracts.

After your answer, list which of the numbered papers above you actually relied on, e.g. "Sources: [1], [3]".

Papers:
{context}

Question: {question}

Answer:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="Natural language question")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--since", type=str, default=None, help="Only consider papers submitted on/after this date")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama model name to use for the answer")
    args = parser.parse_args()

    print("Loading embedding model...")
    embed_model = SentenceTransformer(MODEL_NAME)
    es = Elasticsearch(ES_HOST)

    print("Retrieving relevant papers...")
    papers = retrieve(args.question, embed_model, es, k=args.k, since=args.since)

    print(f"\nRetrieved {len(papers)} papers:")
    for i, p in enumerate(papers, start=1):
        print(f"  [{i}] {p['title']} ({p.get('submitted')})")

    prompt = build_prompt(args.question, papers)

    print(f"\nAsking {args.model}...\n")
    response = ollama.chat(model=args.model, messages=[{"role": "user", "content": prompt}])

    print("=== Answer ===")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
