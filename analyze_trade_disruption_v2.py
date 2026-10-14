"""
Trade Disruption Detection Pipeline v2
-----------------------------------------
Fixes from v1's diagnostic findings:
1. Reindexes each series to the full month range, filling gaps with 0
2. Replaces flat mean/std baseline with a LOG-LINEAR TREND baseline, so
   long-term organic growth isn't mistaken for a disruption anomaly
3. Ground truth restricted to commodities Russia/Belarus actually dominate
   (fertilizers, energy) - not every commodity they trade in

NOTE: This still uses Russia/Belarus's own REPORTER-side data, which is
sparse/absent post-2022 (the real reporting gap). For a stronger
evaluation set, this should be combined with mirror-statistics data
(Russia/Belarus as PARTNER in other countries' reports) - see the
separate mirror-statistics extraction step.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression

# ============ CONFIG ============

HDFS_PATH = "hdfs://localhost:9000/trade_data/trade_data_final_merged.csv"
EVENT_START_YEAR = 2022
EVENT_START_MONTH = 2
Z_SCORE_THRESHOLD = 2.5

DISRUPTED_COUNTRIES = {"Russian Federation", "Belarus"}
# Commodities Russia/Belarus are actually documented to dominate globally -
# restricts ground truth to plausible disruption commodities, not everything
DISRUPTED_COMMODITIES = {
    "Nitrogenous_fertilizers", "Phosphatic_fertilizers", "Potassic_fertilizers",
    "Coal", "Crude_oil", "Natural_gas", "Wheat"
}

# ============ STAGE 1: LOAD + AGGREGATE ============

spark = SparkSession.builder.appName("TradeDisruptionAnalysisV2").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("Reading data from HDFS...")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(HDFS_PATH)

print("Aggregating to monthly (country, commodity, flow) time series...")
monthly = (
    df.groupBy("reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc", "refYear", "refMonth")
    .agg(spark_sum("primaryValue").alias("trade_value"))
)

print("Collecting to pandas...")
pdf = monthly.toPandas()
pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]
spark.stop()

EVENT_START_PERIOD = EVENT_START_YEAR * 100 + EVENT_START_MONTH

# ============ STAGE 2: FILL MISSING MONTHS ============

print("Reindexing each series to the full month range (filling gaps with 0)...")

all_periods = []
for y in range(2017, 2026):
    for m in range(1, 13):
        p = y * 100 + m
        if p <= 202512:
            all_periods.append(p)

group_cols = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]
series_keys = pdf[group_cols].drop_duplicates()

full_index = series_keys.merge(pd.DataFrame({"period": all_periods}), how="cross")
pdf_full = full_index.merge(pdf[group_cols + ["period", "trade_value"]], on=group_cols + ["period"], how="left")
pdf_full["trade_value"] = pdf_full["trade_value"].fillna(0)
pdf_full["refYear"] = pdf_full["period"] // 100
pdf_full["refMonth"] = pdf_full["period"] % 100

print(f"Full reindexed dataset: {len(pdf_full):,} rows across {len(series_keys):,} series.")

# ============ STAGE 3: LOG-LINEAR TREND BASELINE ============

print("Fitting log-linear trend baseline per series...")

results = []

for group_key, group_df in pdf_full.groupby(group_cols):
    group_df = group_df.sort_values("period").reset_index(drop=True)
    group_df["time_idx"] = range(len(group_df))

    baseline = group_df[group_df["period"] < EVENT_START_PERIOD].copy()
    evaluation = group_df[group_df["period"] >= EVENT_START_PERIOD].copy()

    # need a reasonable amount of non-zero baseline data to fit a trend
    if len(baseline) < 12 or (baseline["trade_value"] > 0).sum() < 6:
        continue

    # log1p handles zeros safely; stabilizes variance for skewed trade values
    baseline["log_value"] = np.log1p(baseline["trade_value"])
    group_df["log_value"] = np.log1p(group_df["trade_value"])

    # Fit trend line: log_value ~ time_idx, using only the baseline period
    X_baseline = baseline[["time_idx"]].values
    y_baseline = baseline["log_value"].values
    trend_model = LinearRegression()
    trend_model.fit(X_baseline, y_baseline)

    # Predict the expected trend for the ENTIRE series (baseline + evaluation)
    group_df["trend_pred"] = trend_model.predict(group_df[["time_idx"]].values)
    group_df["residual"] = group_df["log_value"] - group_df["trend_pred"]

    baseline_residuals = group_df.loc[group_df["period"] < EVENT_START_PERIOD, "residual"]
    residual_std = baseline_residuals.std()

    if residual_std == 0 or np.isnan(residual_std):
        continue

    group_df["z_score"] = group_df["residual"] / residual_std
    group_df["is_baseline_period"] = group_df["period"] < EVENT_START_PERIOD
    group_df["z_anomaly"] = (group_df["z_score"].abs() > Z_SCORE_THRESHOLD) & (~group_df["is_baseline_period"])

    results.append(group_df)

full_results = pd.concat(results, ignore_index=True)
print(f"Processed {len(results)} valid series with trend-adjusted baselines.")

# ============ STAGE 4: ISOLATION FOREST (on trend residuals now, not raw z) ============

print("Training Isolation Forest on trend residuals...")

if_predictions = []

for group_key, group_df in full_results.groupby(group_cols):
    group_df = group_df.sort_values("period").copy()
    baseline = group_df[group_df["is_baseline_period"]]
    evaluation = group_df[~group_df["is_baseline_period"]]

    if len(baseline) < 12 or len(evaluation) < 1:
        group_df["if_anomaly"] = False
        if_predictions.append(group_df)
        continue

    features_baseline = baseline[["residual"]].fillna(0)
    features_eval = evaluation[["residual"]].fillna(0)

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features_baseline)

    eval_preds = model.predict(features_eval)
    group_df["if_anomaly"] = False
    group_df.loc[evaluation.index, "if_anomaly"] = eval_preds == -1

    if_predictions.append(group_df)

final = pd.concat(if_predictions, ignore_index=True)

# ============ STAGE 5: VALIDATION (restricted ground truth) ============

print("Validating against restricted ground truth (Russia/Belarus, documented commodities only)...")

eval_only = final[~final["is_baseline_period"]].copy()
eval_only["ground_truth"] = (
    eval_only["reporterDesc"].isin(DISRUPTED_COUNTRIES)
    & eval_only["commodity_name"].isin(DISRUPTED_COMMODITIES)
).astype(int)

print(f"\nEvaluation rows: {len(eval_only):,}")
print(f"Ground truth positive rows: {eval_only['ground_truth'].sum():,}")
print("NOTE: if this number is very small, Russia/Belarus's reporter-side gap")
print("means there isn't enough real data to validate against yet - this is")
print("exactly why the mirror-statistics extraction is needed as a next step.\n")

eval_only["z_pred"] = eval_only["z_anomaly"].astype(int)
eval_only["if_pred"] = eval_only["if_anomaly"].astype(int)

for method, pred_col in [("Z-score (trend-adjusted)", "z_pred"), ("Isolation Forest", "if_pred")]:
    precision = precision_score(eval_only["ground_truth"], eval_only[pred_col], zero_division=0)
    recall = recall_score(eval_only["ground_truth"], eval_only[pred_col], zero_division=0)
    f1 = f1_score(eval_only["ground_truth"], eval_only[pred_col], zero_division=0)
    print(f"{method}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}\n")

# ============ STAGE 6: RANKED RESULTS ============

ranking = (
    eval_only.groupby(["reporterDesc", "commodity_name", "flowDesc"])["z_score"]
    .apply(lambda x: x.abs().mean())
    .reset_index()
    .rename(columns={"z_score": "avg_abs_z_score"})
    .sort_values("avg_abs_z_score", ascending=False)
)

print("Most affected country-commodity pairs (trend-adjusted):")
print(ranking.head(20).to_string(index=False))

final.to_csv("trade_disruption_results_v2.csv", index=False)
ranking.to_csv("trade_disruption_ranking_v2.csv", index=False)

print("\nDone. Results saved to trade_disruption_results_v2.csv and trade_disruption_ranking_v2.csv")
