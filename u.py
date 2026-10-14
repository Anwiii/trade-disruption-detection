import os
from collections import defaultdict

files = os.listdir("trade_data_raw_batched")
group_files = [f for f in files if "_ALLCOMM_" not in f]

coverage = defaultdict(set)
for f in group_files:
    parts = f.replace(".csv", "").split("_")
    country = parts[0]
    coverage[country].add((parts[1], parts[2]))  # (group, year)

for country, combos in sorted(coverage.items()):
    print(f"{country}: {len(combos)}/18 combos")  # 2 groups x 9 years = 18