"""
Single entry point for the whole project: takes one natural-language
question and figures out which of the four capabilities it's actually
asking for, then routes it there -- so a partner/grader can just ask a
question instead of picking a script by hand.

Routing (hybrid, on purpose):
  1. Keyword rules first -- fast, deterministic, easy to defend in Q&A
     ("why did it pick this feature?" -> "it matched the word 'theorem'").
  2. Only if no rule matches, ask the local LLM (Ollama) to classify the
     question into one of the four categories -- more robust to
     paraphrasing that doesn't use an obvious keyword. This is cheap: it's
     a one-word classification, not content generation, so it doesn't
     carry the same hallucination risk as the RAG/advisor/theorem answers.

Note: "rag" is the default/fallback category and covers both plain search
and RAG question-answering (rag_answer.py already prints the retrieved
papers before the answer, so it's a superset of what search.py shows) --
kept as one route to keep this router simple.

Usage:
    python3 ask.py "What are recent approaches to concentration inequalities for random matrices?"
    python3 ask.py "Where is Hoeffding's inequality proved?"
    python3 ask.py "Who works on convex optimization in Israel?"
    python3 ask.py "What's the trend for random graphs?"
"""

import argparse
import re
import subprocess
import sys

import ollama
import pandas as pd

OLLAMA_MODEL = "llama3.1:8b"

THEOREM_PATTERNS = [
    r"\btheorem\b", r"\bprove[sd]?\b", r"\bproof\b",
    r"\blemma\b", r"\bproposition\b", r"\bcorollary\b",
]
ADVISOR_PATTERNS = [
    r"\badvisor\b", r"\bsupervisor\b", r"\bresearcher[s]?\b",
    r"who (works|studies|is working)", r"\bcollaborat", r"\bphd\b",
]
TREND_PATTERNS = [
    r"\btrend", r"how many papers", r"\bgrowth\b", r"growing",
    r"over the years", r"over time", r"\bpopular(ity)?\b",
    # deliberately no bare "statistic" trigger: math.ST ("statistics") is one
    # of our 4 actual categories, so a genuine content question ("approaches
    # to hypothesis testing in statistics") would misroute to trends instead
    # of RAG -- and since a rule match skips the LLM fallback entirely, this
    # would be a silent, unrecoverable misroute rather than a rare glitch.
]

# Same list spark_kpis.py aggregates over -- keep in sync with KEYWORDS there.
TREND_KEYWORDS = [
    "Markov chain", "martingale", "branching process", "random walk",
    "random graph", "graph coloring", "hypothesis testing",
    "linear programming", "convex optimization",
]


def rule_based_route(question):
    q = question.lower()
    if any(re.search(p, q) for p in THEOREM_PATTERNS):
        return "theorem"
    if any(re.search(p, q) for p in ADVISOR_PATTERNS):
        return "advisor"
    if any(re.search(p, q) for p in TREND_PATTERNS):
        return "trend"
    return None


def llm_route(question):
    prompt = f"""Classify the following question into EXACTLY ONE of these four categories, and reply with ONLY the category word, nothing else:

- rag: a general question about the content, approach, or findings of math research papers
- theorem: asking to find, locate, or verify a specific theorem, lemma, or proof
- advisor: asking to find a researcher, advisor, or expert on a topic
- trend: asking about trends, popularity, or paper counts over time

Question: "{question}"

Category:"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"].strip().lower()
    for category in ("rag", "theorem", "advisor", "trend"):
        if category in answer:
            return category
    return "rag"  # safe default if the LLM's reply doesn't parse cleanly


def show_trend(question):
    q = question.lower()
    matched = [kw for kw in TREND_KEYWORDS if kw.lower() in q]
    df = pd.read_csv("data/kpi_keyword_trends.csv")
    if matched:
        df = df[df["keyword"].isin(matched)]
        print(f"Matched tracked keyword(s) in your question: {', '.join(matched)}\n")
    else:
        print("Couldn't match a specific tracked keyword in your question -- showing all tracked trends.")
        print(f"(Tracked keywords: {', '.join(TREND_KEYWORDS)})\n")
    print(df.to_string(index=False))
    print("\nFull tables: data/kpi_category_year.csv, data/kpi_top_authors.csv, data/kpi_keyword_trends.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="Any natural language question about the project")
    args = parser.parse_args()

    route = rule_based_route(args.question)
    reason = "keyword rule"
    if route is None:
        print("No keyword rule matched -- asking the LLM to classify the question...")
        route = llm_route(args.question)
        reason = "LLM classification"

    print(f"-> routed to: {route}  ({reason})\n")

    if route == "theorem":
        subprocess.run([sys.executable, "find_theorem.py", args.question])
    elif route == "advisor":
        subprocess.run([sys.executable, "find_advisor.py", args.question])
    elif route == "trend":
        show_trend(args.question)
    else:
        subprocess.run([sys.executable, "rag_answer.py", args.question])


if __name__ == "__main__":
    main()
