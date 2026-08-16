"""
Writes a small sample of the real filtered dataset, so graders can see the
actual data shape without downloading the full arXiv dump or running the
pipeline themselves.

Run this locally, after filter_math_subset.py has produced the real
data/arxiv_math_subset.jsonl. The output is small enough (a few dozen real
records) to commit to the repository directly.

Usage:
    python3 make_sample.py
    python3 make_sample.py --n 100
"""

import argparse
import json
import random

INPUT_PATH = "data/arxiv_math_subset.jsonl"
OUTPUT_PATH = "data/sample_arxiv_math_subset.jsonl"
DEFAULT_N = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of sample records to write")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for a reproducible sample")
    args = parser.parse_args()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [line for line in f if line.strip()]

    random.seed(args.seed)
    sample = random.sample(records, min(args.n, len(records)))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for line in sample:
            out.write(line if line.endswith("\n") else line + "\n")

    print(f"Wrote {len(sample)} real records (out of {len(records):,} total) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
