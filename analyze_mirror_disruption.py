"""
Mirror-Based Disruption Analysis for Russia & Belarus
---------------------------------------------------------
Since Russia/Belarus stopped reporting their own trade after Feb 2022,
this script reconstructs their trade using MIRROR STATISTICS: their
24 trading partners' own reports, where partnerISO == RUS or BLR.

This gives genuinely continuous 2017-2025 series for Russia/Belarus,
fixing the empty-evaluation-period problem from v1/v2 (where Russia's
own REPORTER-side data barely existed post-2022).

Same methodology as v2: log-linear trend baseline (Jan 2017-Jan 2022),
z-score on residuals, Isolation Forest on residuals.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# ============ CONFIG ============

MIRROR_PATH = "hdfs://localhost:9000/trade_data/mirror_data"
EVENT_START_PERIOD = 202202
Z_SCORE_THRESHOLD = 2.5

# ============ STAGE 1: LOAD MIRROR DATA + AGGREGATE ============

spark = SparkSession.builder.appName("MirrorDisruptionAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("Reading mirror data from HDFS...")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(MIRROR_PATH)

print("Aggregating by (partner=Russia/Belarus, commodity, flow, year, month)...")
# NOTE: flow direction flips from the mirror's perspective - if a partner
# reports "Import from Russia", that's effectively Russia's EXPORT. We keep
# the partner's own flowDesc label here and note the flip in interpretation.
mirror_monthly = (
    df.groupBy("partnerISO", "cmdCode", "commodity_name", "flowDesc", "refYear", "refMonth")
    .agg(spark_sum("primaryValue").alias("trade_value"))
)

pdf = mirror_monthly.toPandas()
pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]
spark.stop()

print(f"Aggregated to {len(pdf):,} monthly mirror observations.")

# ============ STAGE 2: FILL MISSING MONTHS ============

all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13) if y * 100 + m <= 202512]

group_cols = ["partnerISO", "cmdCode", "commodity_name", "flowDesc"]
series_keys = pdf[group_cols].drop_duplicates()
full_index = series_keys.merge(pd.DataFrame({"period": all_periods}), how="cross")
pdf_full = full_index.merge(pdf[group_cols + ["period", "trade_value"]], on=group_cols + ["period"], how="left")
pdf_full["trade_value"] = pdf_full["trade_value"].fillna(0)

print(f"Reindexed to {len(pdf_full):,} rows across {len(series_keys):,} mirror series.")

# ============ STAGE 3: LOG-LINEAR TREND BASELINE + ANOMALY DETECTION ============

print("Fitting trend baselines and detecting anomalies...")

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

    # Isolation Forest
    baseline_feat = group_df.loc[group_df["is_baseline_period"], ["residual"]].fillna(0)
    eval_feat = group_df.loc[~group_df["is_baseline_period"], ["residual"]].fillna(0)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(baseline_feat)
    eval_idx = group_df[~group_df["is_baseline_period"]].index
    group_df["if_anomaly"] = False
    group_df.loc[eval_idx, "if_anomaly"] = model.predict(eval_feat) == -1

    results.append(group_df)

final = pd.concat(results, ignore_index=True)
print(f"\nProcessed {len(results)} valid Russia/Belarus mirror series.")

eval_only = final[~final["is_baseline_period"]]
print(f"Evaluation-period rows (real data, via mirror statistics): {len(eval_only):,}")
print(f"Anomalies flagged by z-score: {eval_only['z_anomaly'].sum():,}")
print(f"Anomalies flagged by Isolation Forest: {eval_only['if_anomaly'].sum():,}")

# ============ STAGE 4: RANKING ============

ranking = (
    eval_only.groupby(["partnerISO", "commodity_name", "flowDesc"])["z_score"]
    .apply(lambda x: x.abs().mean())
    .reset_index()
    .rename(columns={"z_score": "avg_abs_z_score"})
    .sort_values("avg_abs_z_score", ascending=False)
)

print("\nMost anomalous Russia/Belarus mirror series (by trading partners' own reports):")
print(ranking.head(20).to_string(index=False))

final.to_csv("mirror_disruption_results.csv", index=False)
ranking.to_csv("mirror_disruption_ranking.csv", index=False)

print("\nDone. Saved mirror_disruption_results.csv and mirror_disruption_ranking.csv")
print("\nNEXT STEP: use these real, populated Russia/Belarus evaluation results")
print("as your ground truth case study, instead of the near-empty reporter-side")
print("data - this replaces the flawed validation from v1/v2.")
