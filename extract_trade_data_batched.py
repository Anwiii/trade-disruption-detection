"""
UN Comtrade Bulk Data Extraction Script (OPTIMIZED - batched commodities)
---------------------------------------------------------------------------
Same as the original script, but batches ALL commodities into a single
API call per country-year (using comma-separated cmdCode), instead of
one call per country-commodity-year.

This cuts total calls from ~2,700 down to ~225 (25 countries x 9 years),
meaning your full dataset could finish in well under one day instead
of ~6 days.

IMPORTANT: Test this on 1-2 countries first before running the full
list, to confirm multi-commodity cmdCode actually works as expected
and that row counts don't hit the 100,000 per-call cap.
"""

import comtradeapicall as ca
import pandas as pd
import time
import os
import json
import socket


def has_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

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

COMMODITY_LIST = list(COMMODITIES.values())
COMMODITY_BATCHES = {
    "GroupA": ",".join(COMMODITY_LIST[:6]),   # Wheat, Maize, Sunflower_oil, Soybeans, Nitrogenous, Phosphatic
    "GroupB": ",".join(COMMODITY_LIST[6:]),   # Potassic, Coal, Crude_oil, Natural_gas, Aluminum, Nickel
}

YEARS = list(range(2017, 2026))  # 2017 to 2025

OUTPUT_DIR = "trade_data_raw_batched"
PROGRESS_FILE = "extraction_progress_batched.json"

CALLS_PER_DAY_LIMIT = 480
SLEEP_BETWEEN_CALLS = 0.25

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
    for group_name, commodity_codes in COMMODITY_BATCHES.items():
        for year in YEARS:

            key = f"{reporter_name}_{group_name}_{year}"

            if key in completed:
                continue

            if calls_made_today >= CALLS_PER_DAY_LIMIT:
                print(f"\nReached daily call limit ({CALLS_PER_DAY_LIMIT}). Stopping for today.")
                save_progress()
                exit(0)

            if not has_internet():
                print(f"\nNo internet connection detected. Waiting 30 seconds before retrying...")
                while not has_internet():
                    time.sleep(30)
                    print("Still no connection, waiting...")
                print("Connection restored, continuing.\n")

            try:
                print(f"Pulling: {reporter_name} | {group_name} | {year} ...", end=" ")

                period_str = ",".join(f"{year}{m:02d}" for m in range(1, 13))

                df = ca.getFinalData(
                    subscription_key=SUBSCRIPTION_KEY,
                    typeCode='C',
                    freqCode='M',
                    clCode='HS',
                    period=period_str,
                    reporterCode=reporter_code,
                    cmdCode=commodity_codes,
                    flowCode='M,X',
                    partnerCode=None,
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
                    if len(df) >= 99000:
                        print(f"  WARNING: still near the 100,000-row cap — {reporter_name} {year} {group_name} may need further splitting.")
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

print("\nAll combinations completed! Check the trade_data_raw_batched/ folder for your CSV files.")