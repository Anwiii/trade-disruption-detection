"""
UN Comtrade Bulk Data Extraction Script
-----------------------------------------
Pulls monthly trade data for multiple reporter countries, commodities,
and years. Designed to run across multiple sessions/days since the
free tier caps you at 500 calls/day and 100,000 records/call.

Progress is logged so you can stop and resume without re-pulling
completed combinations.
"""

import comtradeapicall as ca
import pandas as pd
import time
import os
import json

# ============ CONFIG ============


SUBSCRIPTION_KEY = os.environ["COMTRADE_KEY"]

REPORTERS = {
    "Russia": 643, "Ukraine": 804, "Belarus": 112, "USA": 842,
    "Brazil": 76, "Canada": 124, "Australia": 36, "India": 699,
    "China": 156, "Egypt": 818, "Turkiye": 792, "Germany": 276,
    "France": 251, "Indonesia": 360, "Argentina": 32, "Kazakhstan": 398,
    "Nigeria": 566, "Bangladesh": 50, "Pakistan": 586, "Saudi Arabia": 682,
    "South Africa": 710, "Vietnam": 704, "Mexico": 484, "Japan": 392,
    "South Korea": 410,
}

COMMODITIES = {
    "Wheat": "1001",
    "Maize": "1005",
    "Sunflower_oil": "1512",
    "Soybeans": "1201",
    "Nitrogenous_fertilizers": "3102",
    "Phosphatic_fertilizers": "3103",
    "Potassic_fertilizers": "3104",
    "Coal": "2701",
    "Crude_oil": "2709",
    "Natural_gas": "2711",
    "Aluminum": "7601",
    "Nickel": "7502",
}

YEARS = list(range(2017, 2026))  # 2017 to 2025

OUTPUT_DIR = "trade_data_raw"
PROGRESS_FILE = "extraction_progress.json"

CALLS_PER_DAY_LIMIT = 480  # a little under 500 to leave safety margin
SLEEP_BETWEEN_CALLS = 0.25  # seconds, respects 5 calls/sec limit

# ============ SETUP ============

os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        completed = set(json.load(f))
else:
    completed = set()


def save_progress():
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(completed), f)


# ============ MAIN EXTRACTION LOOP ============

calls_made_today = 0

for reporter_name, reporter_code in REPORTERS.items():
    for commodity_name, hs_code in COMMODITIES.items():
        for year in YEARS:

            key = f"{reporter_name}_{commodity_name}_{year}"

            if key in completed:
                continue  # already pulled this combination, skip

            if calls_made_today >= CALLS_PER_DAY_LIMIT:
                print(f"\nReached daily call limit ({CALLS_PER_DAY_LIMIT}). "
                      f"Stopping for today. Resume by running the script again tomorrow.")
                save_progress()
                exit(0)

            try:
                print(f"Pulling: {reporter_name} | {commodity_name} | {year} ...", end=" ")

                # Monthly frequency requires YYYYMM period codes, not just the year
                period_str = ",".join(f"{year}{m:02d}" for m in range(1, 13))

                df = ca.getFinalData(
                    subscription_key=SUBSCRIPTION_KEY,
                    typeCode='C',
                    freqCode='M',              # Monthly
                    clCode='HS',
                    period=period_str,         # e.g. "201701,201702,...,201712"
                    reporterCode=reporter_code,
                    cmdCode=hs_code,
                    flowCode='M,X',            # Imports and Exports
                    partnerCode=None,          # None = all partners
                    partner2Code=None,
                    customsCode=None,
                    motCode=None,
                    maxRecords=100000,
                    format_output='JSON',
                    countOnly=None,
                    includeDesc=True
                )

                if df is not None and len(df) > 0:
                    filename = f"{OUTPUT_DIR}/{key}.csv"
                    df.to_csv(filename, index=False)
                    print(f"OK ({len(df)} rows)")
                else:
                    print("No data returned (empty result, not an error)")

                completed.add(key)
                calls_made_today += 1
                save_progress()
                time.sleep(SLEEP_BETWEEN_CALLS)

            except Exception as e:
                print(f"ERROR: {e}")
                print("Stopping so you can check the issue. Progress saved — rerun to resume.")
                save_progress()
                exit(1)

print("\nAll combinations completed! Check the trade_data_raw/ folder for your CSV files.")
