"""
Cleanup script: removes entries from extraction_progress.json that were
marked "completed" but have no actual CSV file — these were likely
false completions caused by a network outage during extraction.

Run this ONCE before resuming extraction after a connection drop.
"""

import json
import os

PROGRESS_FILE = "extraction_progress.json"
OUTPUT_DIR = "trade_data_raw"

with open(PROGRESS_FILE, "r") as f:
    completed = set(json.load(f))

existing_files = set(f.replace(".csv", "") for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv"))

# Keep only entries that either have a real CSV, OR were confirmed empty
# BEFORE today (we can't fully distinguish legit-empty from network-fail,
# so we conservatively remove ALL entries without a CSV — this means a
# few genuinely-empty combinations like Russia 2022+ will be re-queried,
# which just costs a handful of extra calls, not a real problem)

before_count = len(completed)
cleaned = completed & existing_files
removed = completed - cleaned

with open(PROGRESS_FILE, "w") as f:
    json.dump(list(cleaned), f)

print(f"Before: {before_count} completed entries")
print(f"After:  {len(cleaned)} completed entries")
print(f"Removed (will be retried): {len(removed)}")
print("\nSample of removed entries (first 20):")
for r in list(removed)[:20]:
    print(f"  - {r}")
