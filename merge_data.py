"""
Merge script: combines all CSVs in trade_data_raw_batched/ into one
final combined dataset, ready for HDFS/Spark.

Since trade_data_raw_batched/ fully covers all 25 countries x 12
commodities x 9 years (verified clean via cleanup_progress_batched.py),
this is your complete dataset - the earlier trade_data_raw/ folder
(from before batching) is now redundant and can be left alone.
"""

import pandas as pd
import os
import glob

INPUT_DIR = "trade_data_raw_batched"
OUTPUT_FILE = "trade_data_final_merged.csv"

# Maps HS codes back to friendly commodity names for a readable column
HS_TO_NAME = {
    "1001": "Wheat",
    "1005": "Maize",
    "1512": "Sunflower_oil",
    "1201": "Soybeans",
    "3102": "Nitrogenous_fertilizers",
    "3103": "Phosphatic_fertilizers",
    "3104": "Potassic_fertilizers",
    "2701": "Coal",
    "2709": "Crude_oil",
    "2711": "Natural_gas",
    "7601": "Aluminum",
    "7502": "Nickel",
}

print("Reading all CSV files from", INPUT_DIR, "...")

all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
print(f"Found {len(all_files)} files to merge.")

dataframes = []
for i, filepath in enumerate(all_files, 1):
    try:
        df = pd.read_csv(filepath)
        if len(df) > 0:
            dataframes.append(df)
    except Exception as e:
        print(f"  WARNING: could not read {filepath}: {e}")
    if i % 50 == 0:
        print(f"  ...read {i}/{len(all_files)} files")

print(f"\nSuccessfully read {len(dataframes)} non-empty files.")
print("Combining into one dataset...")

merged = pd.concat(dataframes, ignore_index=True)

# Add a friendly commodity name column based on cmdCode, if that column exists
cmd_col = None
for candidate in ["cmdCode", "commodityCode", "cmdcode"]:
    if candidate in merged.columns:
        cmd_col = candidate
        break

if cmd_col:
    merged["commodity_name"] = merged[cmd_col].astype(str).map(HS_TO_NAME)
else:
    print("  NOTE: could not find a commodity code column to map friendly names - check your columns list below.")

print(f"\nFinal merged dataset: {len(merged):,} rows, {len(merged.columns)} columns")
print("\nColumns:", list(merged.columns))

print("\nSaving to", OUTPUT_FILE, "...")
merged.to_csv(OUTPUT_FILE, index=False)
print("Done.")

# Quick sanity summary
print("\n--- Sanity check ---")
reporter_col = "reporterDesc" if "reporterDesc" in merged.columns else None
year_col = "refYear" if "refYear" in merged.columns else ("period" in merged.columns and "period")

if reporter_col:
    print(f"Unique reporter countries: {merged[reporter_col].nunique()}")
if cmd_col:
    print(f"Unique commodity codes: {merged[cmd_col].nunique()}")
print(f"Total rows: {len(merged):,}")
