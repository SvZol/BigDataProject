"""
Spark job: computes KPIs and topic-trend insights from the filtered arXiv math
subset, enriched with citation counts fetched from Semantic Scholar
(data/citations.jsonl).

This is the "meaningful transformation" step of the pipeline: it reads the
raw filtered records, joins them with citation data, and aggregates into
three small result tables (written as single CSV files, easy to chart for
the presentation):

  1. data/kpi_category_year.csv   -- paper count per category, per year
  2. data/kpi_top_authors.csv     -- most prolific authors in the subset
  3. data/kpi_keyword_trends.csv  -- per keyword, per year: how many papers
                                      mention it, and their average citation
                                      count (topic popularity vs. real impact)

Usage:
    python3 spark_kpis.py
    python3 spark_kpis.py --categories math.CO math.OC

--categories lets you focus the report on a subset of the four indexed
categories (math.PR, math.ST, math.CO, math.OC). It can only select among
these four -- they're the only categories present in the underlying dataset
(and in the Elasticsearch index / RAG assistant), so picking a category
outside this set would silently return nothing. Widening to arbitrary arXiv
categories would require re-running the whole pipeline (filter -> Kafka ->
embeddings -> index) from the 5GB raw dump for the new category, which is a
much bigger step -- out of scope here to keep search/RAG/KPIs consistent
with each other.
"""

import argparse

import pandas as pd
from pyspark.sql import SparkSession, functions as F

DATA_PATH = "data/arxiv_math_from_kafka.jsonl"  # written by kafka_consumer_to_file.py
CITATIONS_PATH = "data/citations.jsonl"
TARGET_CATEGORIES = ["math.PR", "math.ST", "math.CO", "math.OC"]

# Bachelor's-level topics (undergrad probability/combinatorics/optimization
# coursework), so results are easy to defend in Q&A -- not research-level
# jargon like "concentration inequality" or "optimal transport".
KEYWORDS = [
    "Markov chain",
    "martingale",
    "branching process",
    "random walk",
    "random graph",
    "graph coloring",
    "hypothesis testing",
    "linear programming",
    "convex optimization",
]

# below this many papers in a year, an average (e.g. avg_citations) is too
# noisy to trust -- one or two data points isn't a "trend"
MIN_RELIABLE_COUNT = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories", nargs="+", default=TARGET_CATEGORIES, choices=TARGET_CATEGORIES,
        help="Which of the four indexed categories to include in the report (default: all four)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    categories = args.categories
    # tag output filenames when reporting on a subset, so a focused run never
    # overwrites the full four-category report
    suffix = "" if set(categories) == set(TARGET_CATEGORIES) else "_" + "_".join(c.split(".")[-1] for c in categories)
    print(f"Reporting on categories: {', '.join(categories)}")

    spark = SparkSession.builder.appName("arxiv-math-kpis").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    papers = spark.read.json(DATA_PATH)
    citations = spark.read.json(CITATIONS_PATH)

    papers = papers.withColumn("submitted_date", F.to_date("submitted", "yyyy-MM-dd"))
    papers = papers.withColumn("year", F.year("submitted_date"))
    papers = papers.withColumn("category_list", F.split(F.col("categories"), " "))
    papers = papers.withColumn("text", F.lower(F.concat_ws(" ", F.col("title"), F.col("abstract"))))

    # A handful of "submitted" dates may fail to parse into a real date, which
    # leaves "year" null for those rows. If left in, a single null forces every
    # value in the (otherwise all-integer) "year" column to be displayed as a
    # float once it goes through pandas/CSV (e.g. 2023 -> 2023.0) -- so drop
    # them here, before any aggregation, and report how many were dropped.
    total_count = papers.count()
    papers = papers.filter(F.col("year").isNotNull())
    dropped = total_count - papers.count()
    if dropped:
        print(f"Dropped {dropped} paper(s) with an unparseable submission date")

    enriched = papers.join(citations, on="id", how="left")
    # keep only papers that carry at least one of the selected categories, so
    # every downstream table (category/year, authors, keyword trends) reflects
    # the same focused selection
    enriched = enriched.filter(F.arrays_overlap(F.col("category_list"), F.array([F.lit(c) for c in categories])))
    enriched.cache()

    # 1. papers per category per year
    category_year = (
        enriched
        .withColumn("category", F.explode("category_list"))
        .filter(F.col("category").isin(categories))
        .groupBy("category", "year")
        .count()
        .orderBy("category", "year")
    )
    category_year.toPandas().to_csv(f"data/kpi_category_year{suffix}.csv", index=False)
    print(f"Wrote data/kpi_category_year{suffix}.csv")

    # 2. top authors (simple split on ", " or " and " -- good enough for a
    #    ranked overview, not meant to be perfectly precise)
    top_authors = (
        enriched
        .withColumn("author", F.explode(F.split(F.col("authors"), r",\s*|\s+and\s+")))
        .withColumn("author", F.trim(F.col("author")))
        .filter(F.col("author") != "")
        .groupBy("author")
        .count()
        .orderBy(F.desc("count"))
        .limit(20)
    )
    top_authors_df = top_authors.toPandas()
    # arXiv metadata stores author names with raw LaTeX escapes (e.g. Klav\v{z}ar)
    # -- decode them to normal unicode text for clean presentation slides.
    from pylatexenc.latex2text import LatexNodes2Text
    latex_converter = LatexNodes2Text()
    top_authors_df["author"] = top_authors_df["author"].apply(latex_converter.latex_to_text)
    top_authors_df.to_csv(f"data/kpi_top_authors{suffix}.csv", index=False)
    print(f"Wrote data/kpi_top_authors{suffix}.csv")

    # 3. keyword trends: paper count + average citations, per year
    trend_rows = []
    for kw in KEYWORDS:
        matched = enriched.filter(F.col("text").contains(kw.lower()))
        yearly = (
            matched.groupBy("year")
            .agg(F.count("*").alias("paper_count"), F.avg("citation_count").alias("avg_citations"))
            .orderBy("year")
        )
        for row in yearly.collect():
            trend_rows.append({
                "keyword": kw,
                "year": row["year"],
                "paper_count": row["paper_count"],
                "avg_citations": round(row["avg_citations"], 2) if row["avg_citations"] is not None else None,
                "reliable": row["paper_count"] >= MIN_RELIABLE_COUNT,
            })

    pd.DataFrame(trend_rows).to_csv(f"data/kpi_keyword_trends{suffix}.csv", index=False)
    print(f"Wrote data/kpi_keyword_trends{suffix}.csv")

    spark.stop()


if __name__ == "__main__":
    main()
