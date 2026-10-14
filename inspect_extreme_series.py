import pandas as pd

# Read the results produced by the previous analysis
df = pd.read_csv("trade_disruption_results.csv")

# Select the extreme series
series = df[
    (df["reporterDesc"] == "India") &
    (df["commodity_name"] == "Phosphatic_fertilizers") &
    (df["flowDesc"] == "Import")
].copy()

# Sort chronologically
series = series.sort_values("period")

print("\n========== INDIA - PHOSPHATIC FERTILIZERS - IMPORT ==========")

print("Number of observations:", len(series))

print("\n---------- VALUES ----------")

print(
    series[
        [
            "period",
            "trade_value",
            "z_score",
            "pct_change",
            "is_baseline_period",
            "z_anomaly"
        ]
    ].to_string(index=False)
)

print("\n---------- SUMMARY ----------")

baseline = series[series["is_baseline_period"]]

print("Baseline observations:", len(baseline))
print("Baseline mean:", baseline["trade_value"].mean())
print("Baseline std:", baseline["trade_value"].std())
print("Minimum trade value:", series["trade_value"].min())
print("Maximum trade value:", series["trade_value"].max())

print("\n========== INSPECTION COMPLETE ==========")