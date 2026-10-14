"""
Cleanup for the BATCHED extraction. Removes:
1. Entries with no matching CSV AND not in the confirmed-empty list
   (these are likely network failures marked as complete)
2. Germany entries specifically (they hit the 100K row cap = truncated,
   even though a CSV exists for them - need to force a re-pull with the
   new split-commodity-group approach)

CONFIRMED_EMPTY holds entries we've verified are genuinely empty for a
real reason (e.g. Russia stopped reporting after 2022), so cleanup
stops re-flagging them every time it runs.
"""

import json
import os

PROGRESS_FILE = "extraction_progress_batched.json"
OUTPUT_DIR = "trade_data_raw_batched"

# Entries confirmed genuinely empty for a real reason - not network failures.
# Add to this list as you confirm more (e.g. after checking a country's
# reporting history), so cleanup stops repeatedly re-flagging them.
CONFIRMED_EMPTY = {
    f"Russia_{group}_{year}"
    for group in ["GroupA", "GroupB"]
    for year in [2022, 2023, 2024, 2025]
} | {
    f"Belarus_{group}_{year}"
    for group in ["GroupA", "GroupB"]
    for year in [2023, 2024, 2025]
} | {
    f"Bangladesh_{group}_{year}"
    for group in ["GroupA", "GroupB"]
    for year in range(2017, 2026)
} | {
    f"China_{group}_2025"
    for group in ["GroupA", "GroupB"]
} | {
    f"Vietnam_{group}_{year}"
    for group in ["GroupA", "GroupB"]
    for year in [2024, 2025]
} | {
    f"Nigeria_{group}_{year}"
    for group in ["GroupA", "GroupB"]
    for year in [2017, 2019]
}

with open(PROGRESS_FILE, "r") as f:
    completed = set(json.load(f))

existing_files = set(f.replace(".csv", "") for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv"))

before_count = len(completed)

# Keep entries that either have a real CSV, OR are confirmed genuinely empty
cleaned = {c for c in completed if c in existing_files or c in CONFIRMED_EMPTY}

# ALSO remove OLD-format Germany entries specifically (Germany_ALLCOMM_*),
# NOT the new correct Germany_GroupA_*/Germany_GroupB_* entries
cleaned = {c for c in cleaned if not c.startswith("Germany_ALLCOMM_")}

removed = completed - cleaned

with open(PROGRESS_FILE, "w") as f:
    json.dump(list(cleaned), f)

print(f"Before: {before_count} completed entries")
print(f"After:  {len(cleaned)} completed entries")
print(f"Removed (will be retried): {len(removed)}")
print("\nSample of removed entries (first 25):")
for r in list(removed)[:25]:
    print(f"  - {r}")