"""
Trade Disruption Detection Pipeline
-------------------------------------
Stage 1: Aggregate raw trade records to monthly (country, commodity, flow) time series
Stage 2: Compute baseline (pre-war) statistics and z-scores
Stage 3: Train Isolation Forest on baseline patterns, flag anomalies in the eval period
Stage 4: Validate against a known-disruption proxy ground truth (Russia/Belarus post-2022)
Stage 5: Output ranked results (most/least affected country-commodity pairs)

Run this with Spark (Java 17 active) pointed at your HDFS file.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, concat_ws
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# ============ CONFIG ============

HDFS_PATH = "hdfs://localhost:9000/trade_data/trade_data_final_merged.csv"
EVENT_START_YEAR = 2022
EVENT_START_MONTH = 2  # February 2022 - Russia-Ukraine war start
Z_SCORE_THRESHOLD = 2.5

# Countries with confirmed, documented real disruption (used as ground truth proxy)
DISRUPTED_COUNTRIES = {"Russian Federation", "Belarus"}

# ============ STAGE 1: LOAD + AGGREGATE ============

spark = SparkSession.builder.appName("TradeDisruptionAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("Reading data from HDFS...")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(HDFS_PATH)

print("Aggregating to monthly (country, commodity, flow) time series...")
monthly = (
    df.groupBy("reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc", "refYear", "refMonth")
    .agg(spark_sum("primaryValue").alias("trade_value"))
    .orderBy("reporterISO", "cmdCode", "flowDesc", "refYear", "refMonth")
)

print(f"Aggregated to {monthly.count():,} monthly series-points.")

# Bring to pandas for time-series/ML work - this is now small enough (a few hundred
# thousand rows at most: ~600 series x up to 108 months) for a single machine.
print("Collecting aggregated data to pandas...")
pdf = monthly.toPandas()
pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]  # e.g. 202203

spark.stop()

# ============ STAGE 2: BASELINE + Z-SCORE ============

print("Computing baselines and z-scores per series...")

EVENT_START_PERIOD = EVENT_START_YEAR * 100 + EVENT_START_MONTH

results = []

group_cols = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]
for group_keys, group_df in pdf.groupby(group_cols):
    group_df = group_df.sort_values("period").copy()

    baseline = group_df[group_df["period"] < EVENT_START_PERIOD]
    evaluation = group_df[group_df["period"] >= EVENT_START_PERIOD]

    if len(baseline) < 6 or len(evaluation) < 1:
        continue  # not enough data to establish a meaningful baseline

    baseline_mean = baseline["trade_value"].mean()
    baseline_std = baseline["trade_value"].std()

    if baseline_std == 0 or np.isnan(baseline_std):
        continue

    group_df["z_score"] = (group_df["trade_value"] - baseline_mean) / baseline_std
    group_df["pct_change"] = group_df["trade_value"].pct_change()
    group_df["is_baseline_period"] = group_df["period"] < EVENT_START_PERIOD
    group_df["z_anomaly"] = (group_df["z_score"].abs() > Z_SCORE_THRESHOLD) & (~group_df["is_baseline_period"])

    results.append(group_df)

full_results = pd.concat(results, ignore_index=True)
print(f"Processed {len(results)} valid country-commodity-flow series.")

# ============ STAGE 3: ISOLATION FOREST ============

print("Training Isolation Forest per series (on baseline, predicting on evaluation)...")

if_predictions = []

for group_keys, group_df in full_results.groupby(group_cols):
    group_df = group_df.sort_values("period").copy()
    baseline = group_df[group_df["is_baseline_period"]]
    evaluation = group_df[~group_df["is_baseline_period"]]

    if len(baseline) < 6 or len(evaluation) < 1:
        group_df["if_anomaly"] = False
        if_predictions.append(group_df)
        continue

    features_baseline = baseline[["z_score", "pct_change"]].fillna(0)
    features_eval = evaluation[["z_score", "pct_change"]].fillna(0)

    model = IsolationForest(contamination=0.15, random_state=42)
    model.fit(features_baseline)

    eval_preds = model.predict(features_eval)  # -1 = anomaly, 1 = normal
    group_df["if_anomaly"] = False
    group_df.loc[evaluation.index, "if_anomaly"] = eval_preds == -1

    if_predictions.append(group_df)

final = pd.concat(if_predictions, ignore_index=True)

# ============ STAGE 4: VALIDATION AGAINST GROUND TRUTH PROXY ============

print("Validating against known-disruption ground truth (Russia/Belarus, post-Feb 2022)...")

eval_only = final[~final["is_baseline_period"]].copy()
eval_only["ground_truth"] = (
    eval_only["reporterDesc"].isin(DISRUPTED_COUNTRIES)
).astype(int)

eval_only["z_pred"] = eval_only["z_anomaly"].astype(int)
eval_only["if_pred"] = eval_only["if_anomaly"].astype(int)

for method, pred_col in [("Z-score method", "z_pred"), ("Isolation Forest", "if_pred")]:
    precision = precision_score(eval_only["ground_truth"], eval_only[pred_col], zero_division=0)
    recall = recall_score(eval_only["ground_truth"], eval_only[pred_col], zero_division=0)
    f1 = f1_score(eval_only["ground_truth"], eval_only[pred_col], zero_division=0)
    print(f"\n{method}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

# ============ STAGE 5: RANKED RESULTS ============

print("\nMost affected country-commodity pairs (by average |z-score| during evaluation period):")

ranking = (
    eval_only.groupby(["reporterDesc", "commodity_name", "flowDesc"])["z_score"]
    .apply(lambda x: x.abs().mean())
    .reset_index()
    .rename(columns={"z_score": "avg_abs_z_score"})
    .sort_values("avg_abs_z_score", ascending=False)
)

print(ranking.head(20).to_string(index=False))

# Save full results for your report/visualizations
final.to_csv("trade_disruption_results.csv", index=False)
ranking.to_csv("trade_disruption_ranking.csv", index=False)

print("\nDone. Results saved to trade_disruption_results.csv and trade_disruption_ranking.csv")