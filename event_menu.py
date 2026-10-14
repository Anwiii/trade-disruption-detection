"""
event_menu.py
-------------
Presents a menu of known major global disruption events. The user picks
one (or enters a custom date), and the script runs the same trend-based
anomaly detection as analyze_event.py, then generates a chart for the
single most-affected series it finds - no manual date entry required.

Uses the same cached monthly data as analyze_event.py (run
prepare_monthly_data.py first if you haven't already).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

MONTHLY_CACHE = "monthly_series_cache.csv"
Z_SCORE_THRESHOLD = 2.5

# ============ KNOWN EVENTS MENU ============
# Add more here as you like - just needs a name and a YYYY-MM date.

KNOWN_EVENTS = [
    {"name": "Russia-Ukraine War", "date": "2022-02", "note": "Invasion began Feb 24, 2022"},
    {"name": "COVID-19 Pandemic / Export Restrictions", "date": "2020-03", "note": "Global lockdowns, food export bans"},
    {"name": "Suez Canal Blockage (Ever Given)", "date": "2021-03", "note": "6-day blockage of the canal"},
    {"name": "US-China Trade War Escalation", "date": "2018-07", "note": "First major tariff round"},
    {"name": "Custom date (enter your own)", "date": None, "note": ""},
]


def period_to_date(period):
    return datetime(int(period) // 100, int(period) % 100, 1)


# ============ MENU ============

print("=" * 60)
print("TRADE DISRUPTION EVENT EXPLORER")
print("=" * 60)
print("\nSelect a known global event, or enter a custom date:\n")

for i, e in enumerate(KNOWN_EVENTS, 1):
    date_str = e["date"] if e["date"] else "----"
    print(f"  {i}. {e['name']:45s} [{date_str}]  {e['note']}")

choice = int(input("\nEnter a number: ").strip())
selected = KNOWN_EVENTS[choice - 1]

if selected["date"]:
    event_input = selected["date"]
    print(f"\nSelected: {selected['name']} ({event_input})")
else:
    event_input = input("\nEnter the event date (YYYY-MM): ").strip()

event_year, event_month = map(int, event_input.split("-"))
EVENT_START_PERIOD = event_year * 100 + event_month

country_input = input("Enter a country name (or 'all'): ").strip()
commodity_input = input("Enter a commodity name (or 'all'): ").strip()

# ============ LOAD + FILTER DATA ============

pdf = pd.read_csv(MONTHLY_CACHE)
pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]

if country_input.lower() != "all":
    pdf = pdf[pdf["reporterDesc"].str.lower() == country_input.lower()]
if commodity_input.lower() != "all":
    pdf = pdf[pdf["commodity_name"].str.lower() == commodity_input.lower()]

if len(pdf) == 0:
    print("No data found for that filter.")
    exit(0)

all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13) if y * 100 + m <= 202512]
group_cols = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]
series_keys = pdf[group_cols].drop_duplicates()
full_index = series_keys.merge(pd.DataFrame({"period": all_periods}), how="cross")
pdf_full = full_index.merge(pdf[group_cols + ["period", "trade_value"]], on=group_cols + ["period"], how="left")
pdf_full["trade_value"] = pdf_full["trade_value"].fillna(0)

# ============ TREND BASELINE + ANOMALY DETECTION ============

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

    results.append(group_df)

if not results:
    print("Not enough data around that date for the given filters.")
    exit(0)

final = pd.concat(results, ignore_index=True)
eval_only = final[~final["is_baseline_period"]]

ranking = (
    eval_only.groupby(["reporterDesc", "commodity_name", "flowDesc"])["z_score"]
    .apply(lambda x: x.abs().mean())
    .reset_index(name="avg_abs_z")
    .sort_values("avg_abs_z", ascending=False)
)

print(f"\nTop affected series around {event_input}:")
print(ranking.head(10).to_string(index=False))

# ============ AUTO-CHART THE TOP RESULT ============

top = ranking.iloc[0]
series = final[
    (final["reporterDesc"] == top["reporterDesc"])
    & (final["commodity_name"] == top["commodity_name"])
    & (final["flowDesc"] == top["flowDesc"])
].sort_values("period").copy()

series["date"] = series["period"].apply(period_to_date)
series["trend_value"] = np.expm1(series["trend_pred"])
event_date = period_to_date(EVENT_START_PERIOD)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(series["date"], series["trade_value"], color="#2563eb", linewidth=1.5, label="Actual trade value")
ax.plot(series["date"], series["trend_value"], color="#94a3b8", linewidth=1.5, linestyle="--", label="Expected trend")
anomalies = series[series["z_anomaly"] == True]
ax.scatter(anomalies["date"], anomalies["trade_value"], color="#dc2626", s=45, zorder=3, label="Flagged anomaly")
ax.axvline(event_date, color="#f59e0b", linestyle=":", linewidth=2, label="Event date")

title_event = selected["name"] if selected["date"] else f"Event on {event_input}"
ax.set_title(f"{top['reporterDesc']} {top['commodity_name']} {top['flowDesc']}\n({title_event})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Trade Value (USD)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()

output_name = f"top_result_{event_input.replace('-', '_')}.png"
plt.savefig(output_name, dpi=150)
plt.close()

print(f"\nChart saved: {output_name}")
print(f"(Top result: {top['reporterDesc']} | {top['commodity_name']} | {top['flowDesc']})")
