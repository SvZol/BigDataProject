"""
Publishes each paper in the filtered arXiv math subset to a Kafka topic, one
message per paper -- simulating how new preprints would arrive in a real
streaming ingestion pipeline (arXiv publishes new papers continuously, so a
production version of this service would keep consuming new submissions
instead of replaying a static snapshot).

Usage:
    python3 kafka_producer.py
"""

import json
import time

from kafka import KafkaProducer

DATA_PATH = "data/arxiv_math_subset.jsonl"
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "arxiv-papers"


def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    sent = 0
    start = time.time()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            producer.send(TOPIC, value=record)
            sent += 1
            if sent % 5000 == 0:
                print(f"published {sent:,} papers...")

    producer.flush()
    producer.close()
    print(f"\nDone. Published {sent:,} papers to topic '{TOPIC}' in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
