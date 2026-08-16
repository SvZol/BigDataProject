"""
"Find an advisor" feature: given one or more research interests, finds
researchers who are currently active (recent matching papers) and impactful
(citations) in that intersection. Where possible, checks affiliation via the
Semantic Scholar Author API and highlights Israeli institutions as a bonus
(the target user studies in Israel) -- but affiliation is a nice-to-have,
not a requirement for a researcher to be recommended.

Affiliation data is self-reported and often missing, so this is deliberately
conservative: a researcher is only labeled "Israel-affiliated" if Semantic
Scholar has an affiliation string matching a known Israeli institution.
Everyone else is still recommended on the merits (recent papers + citations
in the exact topic), just without a location claim -- the LLM prompt
explicitly forbids inventing an affiliation.

Usage:
    python3 find_advisor.py "Markov chains" "random graphs"
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict

import ollama
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_NAME = "arxiv_math"
ES_HOST = "http://localhost:9200"
OLLAMA_MODEL = "llama3.1:8b"

CANDIDATE_PAPERS = 50        # how many semantically-relevant recent papers to scan
TOP_AUTHOR_CANDIDATES = 15   # how many top authors to check affiliation for

PAPER_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=authors"
AUTHOR_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/author/batch"
    "?fields=name,affiliations,hIndex,paperCount,citationCount"
)

ISRAELI_INSTITUTIONS = [
    "technion", "weizmann", "tel aviv university", "hebrew university",
    "bar-ilan", "bar ilan", "ben-gurion", "ben gurion", "university of haifa",
    "reichman", "idc herzliya", "ariel university", "open university of israel",
    "israel institute of technology",
]


def load_citation_lookup():
    lookup = {}
    with open("data/citations.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lookup[rec["id"]] = rec.get("citation_count")
    return lookup


def semantic_search(interests, embed_model, es, k=CANDIDATE_PAPERS):
    query = " and ".join(interests)
    query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()
    resp = es.search(
        index=INDEX_NAME,
        knn={"field": "embedding", "query_vector": query_vector, "k": k, "num_candidates": k * 10},
        size=k,
        source=["id", "title", "submitted"],
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def post_json(url, payload, retries=5):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            print(f"  HTTP error {e.code}: {e.reason}")
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  network error ({e}), retrying...")
            time.sleep(5 * (attempt + 1))
    return None


def is_israeli(affiliations):
    if not affiliations:
        return False
    text = " ".join(affiliations).lower()
    return any(inst in text for inst in ISRAELI_INSTITUTIONS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interests", nargs="+", help="One or more research interests")
    args = parser.parse_args()

    print("Loading model and citation data...")
    embed_model = SentenceTransformer(MODEL_NAME)
    es = Elasticsearch(ES_HOST)
    citation_lookup = load_citation_lookup()

    print(f"Searching for papers on: {', '.join(args.interests)}")
    papers = semantic_search(args.interests, embed_model, es)
    paper_ids = [p["id"] for p in papers]

    print(f"Looking up authors for {len(paper_ids)} papers via Semantic Scholar...")
    paper_author_results = post_json(PAPER_BATCH_URL, {"ids": [f"ARXIV:{i}" for i in paper_ids]}) or []

    author_stats = defaultdict(lambda: {"name": None, "paper_count": 0, "citations": 0})
    for pid, result in zip(paper_ids, paper_author_results):
        if not result:
            continue
        citations = citation_lookup.get(pid) or 0
        for author in result.get("authors", []):
            aid = author.get("authorId")
            if not aid:
                continue
            author_stats[aid]["name"] = author.get("name")
            author_stats[aid]["paper_count"] += 1
            author_stats[aid]["citations"] += citations

    ranked = sorted(author_stats.items(), key=lambda kv: (kv[1]["paper_count"], kv[1]["citations"]), reverse=True)
    top_candidates = ranked[:TOP_AUTHOR_CANDIDATES]

    print(f"Checking affiliations for top {len(top_candidates)} candidates...")
    author_ids = [aid for aid, _ in top_candidates]
    details = post_json(AUTHOR_BATCH_URL, {"ids": author_ids}) or []

    israeli_matches, other_active = [], []
    for (aid, stats), detail in zip(top_candidates, details):
        if not detail:
            other_active.append(stats)
            continue
        affiliations = detail.get("affiliations") or []
        entry = {**stats, "affiliations": affiliations, "h_index": detail.get("hIndex")}
        (israeli_matches if is_israeli(affiliations) else other_active).append(entry)

    print(f"\nConfirmed Israel-affiliated candidates: {len(israeli_matches)}")
    for a in israeli_matches:
        print(f"  {a['name']} -- {a['affiliations']} "
              f"(papers: {a['paper_count']}, citations: {a['citations']}, h-index: {a.get('h_index')})")

    print(f"\nOther active researchers, affiliation not confirmed as Israeli: {len(other_active)}")
    for a in other_active[:5]:
        print(f"  {a['name']} (papers: {a['paper_count']}, citations: {a['citations']})")

    context_lines = []
    if israeli_matches:
        context_lines.append("Researchers CONFIRMED to be affiliated with an Israeli institution:")
        for a in israeli_matches:
            context_lines.append(
                f"- {a['name']}, affiliation: {a['affiliations']}, {a['paper_count']} recent matching papers, "
                f"{a['citations']} total citations on those papers, h-index {a.get('h_index')}"
            )
    else:
        context_lines.append("No candidate researchers had a confirmed Israeli institutional affiliation in the available data.")

    if other_active:
        context_lines.append(
            "\nOther researchers active in this exact topic, affiliation NOT confirmed as Israeli "
            "(do not claim they are in Israel):"
        )
        for a in other_active[:5]:
            context_lines.append(f"- {a['name']}, {a['paper_count']} recent matching papers, {a['citations']} total citations")

    context = "\n".join(context_lines)
    prompt = f"""A student is looking for potential thesis advisors or collaborators interested in: {', '.join(args.interests)}.

Based ONLY on the data below, recommend the most active and well-cited researchers in this exact intersection. If any of them are CONFIRMED to be affiliated with an Israeli institution, mention that clearly (the student is based in Israel, so this is a nice bonus, not a requirement). For everyone else, present them as active, relevant researchers whose institution is simply not confirmed in our data -- never guess or invent an affiliation for them, and don't apologize for not finding an Israeli match; just present the researchers on their merits.

{context}

Write a short, friendly recommendation."""

    print(f"\nAsking {OLLAMA_MODEL} for a recommendation...\n")
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])

    print("=== Recommendation ===")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
