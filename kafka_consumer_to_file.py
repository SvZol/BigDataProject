"""
Consumes all papers currently sitting in the Kafka topic and writes them to
a local file, ready for the Spark KPI job. Reads from the beginning of the
topic and stops once it has caught up (a short idle timeout ends the batch
pickup, since this is a one-shot consumer rather than an always-on service).

Usage:
    python3 kafka_consumer_to_file.py
"""

import json

from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "arxiv-papers"
OUTPUT_PATH = "data/arxiv_math_from_kafka.jsonl"
IDLE_TIMEOUT_MS = 10_000  # stop after 10s with no new messages


def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=IDLE_TIMEOUT_MS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for message in consumer:
            out.write(json.dumps(message.value) + "\n")
            count += 1
            if count % 5000 == 0:
                print(f"consumed {count:,} papers...")

    consumer.close()
    print(f"\nDone. Consumed {count:,} papers from topic '{TOPIC}' -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
