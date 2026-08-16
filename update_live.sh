#!/bin/bash
# Runs the live-update pipeline once: checks the real arXiv API for papers
# newer than what we have, publishes them to Kafka, and indexes them into
# Elasticsearch. Meant to be triggered on a schedule (see launchd/README.md),
# but also safe to run by hand any time:
#
#   bash update_live.sh
#
# Requires Docker (Elasticsearch + Kafka) to already be running. If
# Elasticsearch is not reachable, this exits quietly instead of failing
# loudly, so a scheduled run on a day the containers are stopped just skips
# and logs it, rather than erroring on every trigger.

cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/update_live_$(date +%Y-%m-%d_%H-%M-%S).log"

{
  echo "=== Live update run: $(date) ==="

  if ! curl -s -o /dev/null -m 5 http://localhost:9200; then
    echo "Elasticsearch is not reachable at localhost:9200. Is Docker running? Skipping this run."
    exit 0
  fi

  python3 live_updates/poll_arxiv_live.py
  python3 live_updates/consume_arxiv_live.py

  echo "=== Done: $(date) ==="
} >> "$LOG_FILE" 2>&1

echo "Live update finished, see $LOG_FILE"
