"""
STEP 2 (run anytime, as many times as you want): Interactively analyze
ANY event date, optionally filtered to a specific country/commodity.

Uses the local cache from prepare_monthly_data.py - no Spark/HDFS needed,
so this runs in seconds regardless of how many times you re-analyze.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

MONTHLY_CACHE = "monthly_series_cache.csv"
Z_SCORE_THRESHOLD = 2.5

# ============ GET USER INPUT ============

print("=" * 60)
print("TRADE DISRUPTION ANALYSIS - Custom Event Lookup")
print("=" * 60)

event_input = input("\nEnter the event date (YYYY-MM), e.g. 2022-02: ").strip()
event_year, event_month = map(int, event_input.split("-"))
EVENT_START_PERIOD = event_year * 100 + event_month

country_input = input("Enter a country name (or 'all'): ").strip()
commodity_input = input("Enter a commodity name (or 'all'): ").strip()

print(f"\nAnalyzing disruption around {event_input}...")
print(f"Country filter: {country_input}")
print(f"Commodity filter: {commodity_input}\n")

# ============ LOAD CACHED DATA ============

pdf = pd.read_csv(MONTHLY_CACHE)
pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]

if country_input.lower() != "all":
    pdf = pdf[pdf["reporterDesc"].str.lower() == country_input.lower()]
if commodity_input.lower() != "all":
    pdf = pdf[pdf["commodity_name"].str.lower() == commodity_input.lower()]

if len(pdf) == 0:
    print("No data found matching that country/commodity filter. Check spelling and try again.")
    exit(0)

# ============ FILL MISSING MONTHS ============

all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13) if y * 100 + m <= 202512]
group_cols = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]
series_keys = pdf[group_cols].drop_duplicates()
full_index = series_keys.merge(pd.DataFrame({"period": all_periods}), how="cross")
pdf_full = full_index.merge(pdf[group_cols + ["period", "trade_value"]], on=group_cols + ["period"], how="left")
pdf_full["trade_value"] = pdf_full["trade_value"].fillna(0)

# ============ TREND BASELINE + ANOMALY DETECTION ============
# Baseline = everything BEFORE the event date the user entered.
# Evaluation = everything FROM the event date onward.
# This is the key difference from before - EVENT_START_PERIOD is
# whatever the user typed in, not a hardcoded date.

results = []

for group_key, group_df in pdf_full.groupby(group_cols):
    group_df = group_df.sort_values("period").reset_index(drop=True)
    group_df["time_idx"] = range(len(group_df))

    baseline = group_df[group_df["period"] < EVENT_START_PERIOD].copy()
    evaluation = group_df[group_df["period"] >= EVENT_START_PERIOD].copy()

    if len(baseline) < 12 or (baseline["trade_value"] > 0).sum() < 6 or len(evaluation) < 1:
        continue

    baseline["log_value"] = np.log1p(baseline["trade_value"])
    group_df["log_value"] = np.log1p(group_df["trade_value"])

    trend_model = LinearRegression()
    trend_model.fit(baseline[["time_idx"]].values, baseline["log_value"].values)

    group_df["trend_pred"] = trend_model.predict(group_df[["time_idx"]].values)
    group_df["residual"] = group_df["log_value"] - group_df["trend_pred"]

    baseline_residuals = group_df.loc[group_df["period"] < EVENT_START_PERIOD, "residual"]
    residual_std = baseline_residuals.std()

    if residual_std == 0 or np.isnan(residual_std):
        continue

    group_df["z_score"] = group_df["residual"] / residual_std
    group_df["is_baseline_period"] = group_df["period"] < EVENT_START_PERIOD
    group_df["z_anomaly"] = (group_df["z_score"].abs() > Z_SCORE_THRESHOLD) & (~group_df["is_baseline_period"])

    baseline_feat = group_df.loc[group_df["is_baseline_period"], ["residual"]].fillna(0)
    eval_feat = group_df.loc[~group_df["is_baseline_period"], ["residual"]].fillna(0)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(baseline_feat)
    eval_idx = group_df[~group_df["is_baseline_period"]].index
    group_df["if_anomaly"] = False
    group_df.loc[eval_idx, "if_anomaly"] = model.predict(eval_feat) == -1

    results.append(group_df)

if not results:
    print("Not enough baseline/evaluation data around that date for the given filters.")
    print("Try a date with more history before it, or broaden your country/commodity filter.")
    exit(0)

final = pd.concat(results, ignore_index=True)
eval_only = final[~final["is_baseline_period"]]

# ============ RESULTS ============

print(f"\nAnalyzed {len(results)} series.")
print(f"Evaluation-period rows: {len(eval_only):,}")
print(f"Anomalies flagged by z-score: {eval_only['z_anomaly'].sum():,}")
print(f"Anomalies flagged by Isolation Forest: {eval_only['if_anomaly'].sum():,}")

ranking = (
    eval_only.groupby(["reporterDesc", "commodity_name", "flowDesc"])["z_score"]
    .apply(lambda x: x.abs().mean())
    .reset_index()
    .rename(columns={"z_score": "avg_abs_z_score"})
    .sort_values("avg_abs_z_score", ascending=False)
)

print(f"\nMost affected series around {event_input}:")
print(ranking.head(20).to_string(index=False))

output_name = f"event_analysis_{event_input.replace('-', '_')}.csv"
final.to_csv(output_name, index=False)
print(f"\nFull results saved to {output_name}")
