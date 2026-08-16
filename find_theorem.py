"""
"Find the proof" feature: given a description of a theorem/result, semantically
searches for the most relevant recent papers (same index as search.py), then
fetches their LaTeX source directly from arXiv (not just the abstract),
extracts theorem/lemma/proposition statements and any proofs, ranks them
against the query, and asks an LLM to point to the specific paper (and
theorem number, when identifiable) that most likely matches.

Citation hop (one hop only): if the matched excerpt itself cites another
paper (e.g. "by Theorem 2.1 of \\cite{Tropp2012}, ..."), we resolve that
citation key against the paper's own bibliography (parsed from the same
LaTeX source) and show the referenced work too -- so if a result is only
*used* in the matched paper but actually proved elsewhere, that's visible
instead of silently attributing it to the wrong paper. We do not follow
citation chains beyond this single hop; each additional hop compounds
uncertainty fast, so it isn't worth pretending it's reliable.

This only fetches full text for a handful of top candidate papers per query
(CANDIDATE_PAPERS) -- not the whole 57k-paper corpus, which would be far too
slow and against arXiv's fair-use guidance for bulk downloading.

Honest limits:
  - Only works for papers in our indexed subset (last 3 years, math.PR/ST/CO/OC).
  - LaTeX/bibliography parsing is heuristic regex, not a real parser -- unusual
    macros or formatting can be missed.
  - The one-hop citation lookup identifies *which* work is cited, not whether
    that work actually contains the proof -- we only fetch its title/abstract
    (via Semantic Scholar), not its full text.
  - If nothing scores as a confident match, the tool says so rather than
    forcing a guess.

Usage:
    python3 find_theorem.py "concentration inequality for the largest eigenvalue of a random matrix"
"""

import argparse
import gzip
import io
import json
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

import ollama
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_NAME = "arxiv_math"
ES_HOST = "http://localhost:9200"
OLLAMA_MODEL = "llama3.1:8b"

CANDIDATE_PAPERS = 8       # how many semantically-relevant papers to fetch full text for
TOP_MATCHES = 5            # how many extracted blocks to show the LLM
FETCH_DELAY_S = 3.0        # be polite to arxiv.org between source downloads
MAX_S2_LOOKUPS = 3         # cap on Semantic Scholar citation-hop lookups per run

THEOREM_ENVS = "theorem|thm|lemma|lem|proposition|prop|corollary|cor|claim"
THEOREM_RE = re.compile(
    r"\\begin\{(" + THEOREM_ENVS + r")\}(\[[^\]]*\])?(.*?)\\end\{\1\}",
    re.DOTALL | re.IGNORECASE,
)
PROOF_RE = re.compile(r"\\begin\{proof\}(.*?)\\end\{proof\}", re.DOTALL | re.IGNORECASE)
CITE_RE = re.compile(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])?\{([^}]+)\}")
BIBITEM_RE = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
    re.DOTALL,
)


def semantic_search(query, embed_model, es, k=CANDIDATE_PAPERS):
    query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()
    resp = es.search(
        index=INDEX_NAME,
        knn={"field": "embedding", "query_vector": query_vector, "k": k, "num_candidates": k * 10},
        size=k,
        source=["id", "title"],
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def fetch_source(arxiv_id):
    """Returns (tex_text, bib_text) -- concatenated .tex files and any .bib files."""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "student-thesis-search-project/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"  could not fetch source: {e}")
        return None, ""

    try:
        tex_chunks, bib_chunks = [], []
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if not f:
                    continue
                content = f.read().decode("utf-8", errors="ignore")
                if member.name.endswith((".tex", ".bbl")):
                    # .bbl is the *compiled* bibliography BibTeX produces --
                    # authors almost always submit this instead of the raw
                    # .bib (their personal reference database), so it's the
                    # one that's actually present on arXiv most of the time.
                    # It contains \bibitem entries, same as inline bibliographies.
                    tex_chunks.append(content)
                elif member.name.endswith(".bib"):
                    bib_chunks.append(content)
        if tex_chunks:
            return "\n".join(tex_chunks), "\n".join(bib_chunks)
    except tarfile.TarError:
        pass

    # some submissions are just a single gzipped .tex file
    try:
        return gzip.decompress(raw).decode("utf-8", errors="ignore"), ""
    except OSError:
        return None, ""  # PDF-only submission or unrecognized format


def extract_blocks(tex):
    """Returns a list of (kind, text) -- theorem-like statements and proofs,
    treated independently rather than trying to pair them (pairing by
    position is unreliable with irregular LaTeX)."""
    blocks = [(env.lower(), stmt.strip()) for env, _, stmt in THEOREM_RE.findall(tex) if stmt.strip()]
    blocks += [("proof", p.strip()) for p in PROOF_RE.findall(tex) if p.strip()]
    return blocks


def parse_bibtex_entries(bib_text):
    """Minimal BibTeX scanner (brace-depth counting, not a real parser)."""
    entries = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text):
        key = m.group(1)
        start = m.end()
        depth = 1
        i = start
        while i < len(bib_text) and depth > 0:
            if bib_text[i] == "{":
                depth += 1
            elif bib_text[i] == "}":
                depth -= 1
            i += 1
        body = bib_text[start:i]
        title_m = re.search(r"title\s*=\s*\{(.*?)\}\s*,", body, re.DOTALL | re.IGNORECASE)
        author_m = re.search(r"author\s*=\s*\{(.*?)\}\s*,", body, re.DOTALL | re.IGNORECASE)
        year_m = re.search(r"year\s*=\s*\{?(\d{4})\}?", body, re.IGNORECASE)
        title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else key
        author = re.sub(r"\s+", " ", author_m.group(1)).strip() if author_m else ""
        year = year_m.group(1) if year_m else ""
        citation = " ".join(x for x in [author, f"({year})." if year else "", title] if x)
        entries[key] = citation.strip() or key
    return entries


def parse_bibliography(tex, bib_text):
    bib = {}
    for key, body in BIBITEM_RE.findall(tex):
        text = re.sub(r"\s+", " ", body).strip()
        bib[key.strip()] = text[:280]
    bib.update(parse_bibtex_entries(bib_text))  # bibtex entries win if both present
    return bib


def find_citation_strings(text, bib_dict):
    keys = []
    for m in CITE_RE.finditer(text):
        keys.extend(k.strip() for k in m.group(1).split(","))
    seen, results = set(), []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        if k in bib_dict:
            results.append(bib_dict[k])
    return results


def semantic_scholar_lookup(query_text):
    """One-hop lookup: try to resolve a citation string to a real paper via
    Semantic Scholar's title search. Best-effort -- returns None on any failure
    rather than raising, since this is a bonus enrichment, not core logic."""
    params = urllib.parse.urlencode({"query": query_text[:300], "fields": "title,abstract,year,externalIds", "limit": 1})
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "student-thesis-search-project/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        papers = data.get("data") or []
        return papers[0] if papers else None
    except Exception as e:
        print(f"    (citation-hop lookup failed: {e})")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Description of the theorem/result you're looking for")
    args = parser.parse_args()

    print("Loading model...")
    embed_model = SentenceTransformer(MODEL_NAME)
    es = Elasticsearch(ES_HOST)

    print(f"Semantic search for candidate papers: {args.query}")
    papers = semantic_search(args.query, embed_model, es)
    query_vec = embed_model.encode(args.query, normalize_embeddings=True)

    all_matches = []  # (score, title, arxiv_id, kind, text, citations)
    for p in papers:
        print(f"Fetching source for {p['id']} ({p['title'][:60]}...)")
        tex, bib_text = fetch_source(p["id"])
        time.sleep(FETCH_DELAY_S)
        if not tex:
            print("  source not available, skipping")
            continue

        blocks = extract_blocks(tex)
        if not blocks:
            print("  no theorem/proof-like environments found")
            continue
        bib_dict = parse_bibliography(tex, bib_text)
        print(f"  found {len(blocks)} candidate blocks, {len(bib_dict)} bibliography entries")

        texts = [b[1] for b in blocks]
        embeds = embed_model.encode(texts, normalize_embeddings=True)
        sims = embeds @ query_vec
        for (kind, text), sim in zip(blocks, sims):
            citations = find_citation_strings(text, bib_dict)
            all_matches.append((float(sim), p["title"], p["id"], kind, text, citations))

    if not all_matches:
        print("\nNo theorem-like statements found in the fetched sources for this query.")
        return

    all_matches.sort(key=lambda x: x[0], reverse=True)
    top_matches = all_matches[:TOP_MATCHES]

    print(f"\nTop {len(top_matches)} candidate matches:")
    for score, title, aid, kind, text, citations in top_matches:
        tag = f" [cites: {'; '.join(citations[:2])}]" if citations else ""
        print(f"  [{score:.3f}] {title} ({aid}) -- \\{kind}{tag}")

    # one-hop citation resolution via Semantic Scholar, capped and best-effort
    s2_lookups_done = 0
    s2_cache = {}
    for _, _, _, _, _, citations in top_matches:
        for c in citations:
            if s2_lookups_done >= MAX_S2_LOOKUPS or c in s2_cache:
                continue
            print(f"Looking up citation via Semantic Scholar: {c[:80]}...")
            result = semantic_scholar_lookup(c)
            s2_cache[c] = result
            s2_lookups_done += 1
            time.sleep(1.5)

    context_blocks = []
    for i, (score, title, aid, kind, text, citations) in enumerate(top_matches, start=1):
        block = f"[{i}] Paper: {title} (arXiv:{aid})\n{kind.capitalize()} text:\n{text[:1200]}"
        if citations:
            block += "\n\nCited within this excerpt (may be the true origin of the result):"
            for c in citations[:2]:
                block += f"\n  - {c}"
                s2 = s2_cache.get(c)
                if s2:
                    ext = s2.get("externalIds") or {}
                    arxiv_ref = f", arXiv:{ext['ArXiv']}" if ext.get("ArXiv") else ""
                    block += f"\n    (matched via Semantic Scholar: \"{s2.get('title')}\"{arxiv_ref})"
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)

    prompt = f"""A student is looking for the source of a specific mathematical result, described as: "{args.query}"

Below are candidate theorem/lemma/proposition/proof excerpts extracted directly from the LaTeX source of recent arXiv papers, ranked by semantic similarity to the description. Some excerpts list a citation found inside them -- if present, that citation may be the *actual* origin of the result (the matched paper may just be using or restating it), so mention this distinction clearly if relevant. Identify which excerpt (if any) actually matches the described result. If none convincingly match, say so honestly instead of forcing a match -- these are automatically extracted and may be irrelevant or incomplete.

{context}

Answer, citing the paper (arXiv id) when you can identify a real match, and flag if the result seems to actually originate from a different, cited work."""

    print(f"\nAsking {OLLAMA_MODEL}...\n")
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])

    print("=== Answer ===")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
