"""
explain_anomalies.py
--------------------
Takes trade anomalies already detected by analyze_event.py (or
analyze_mirror_disruption.py) and looks for POTENTIALLY RELATED real-world
events in the GDELT monthly extract.

For each top anomalous series (country / commodity / flow):
  1. Find its strongest flagged anomaly month.
  2. Take a window of +/- N months around that month.
  3. For that country, find which GDELT event categories became unusually
     common in the window, compared with the country's own earlier normal.
  4. Print and save the top categories.

Important: a matching spike in time is NOT proof of cause. Results should be
described as "potentially related events".
"""

import pandas as pd
import numpy as np

GDELT_FILE = "gdelt_monthly_events.csv"
MIN_BASELINE_COUNT = 20     # ignore categories that are very rare for a country
TOP_CATEGORIES = 5          # categories shown per anomaly
MIN_SPIKE_Z = 3.0           # a category must be this unusual (z-score of its share)...
MIN_TIMES_TYPICAL = 1.5     # ...AND at least this many times its typical monthly count
MIN_PRIOR_MONTHS = 24       # prefer a baseline made only of months BEFORE the window

CAMEO_NAMES = {
    1: "Make public statement", 2: "Appeal", 3: "Express intent to cooperate",
    4: "Consult", 5: "Engage in diplomatic cooperation",
    6: "Engage in material cooperation", 7: "Provide aid", 8: "Yield",
    9: "Investigate", 10: "Demand", 11: "Disapprove", 12: "Reject",
    13: "Threaten", 14: "Protest", 15: "Exhibit force posture",
    16: "Reduce relations (e.g. sanctions, boycotts)",
    17: "Coerce (e.g. restrictions, seizures)", 18: "Assault", 19: "Fight",
    20: "Use unconventional mass violence",
}

EXPECTED_COUNTRIES = [
    "RUS", "UKR", "BLR", "USA", "BRA", "CAN", "AUS", "IND", "CHN", "EGY",
    "TUR", "DEU", "FRA", "IDN", "ARG", "KAZ", "NGA", "BGD", "PAK", "SAU",
    "ZAF", "VNM", "MEX", "JPN", "KOR",
]


def month_index(period):
    return (period // 100) * 12 + (period % 100 - 1)


def period_str(period):
    return f"{int(period) // 100}-{int(period) % 100:02d}"


# ============ LOAD + CHECK GDELT EXTRACT ============

print("Loading GDELT extract...")
g = pd.read_csv(GDELT_FILE)
g["EventRootCode"] = pd.to_numeric(g["EventRootCode"], errors="coerce")
g = g.dropna(subset=["EventRootCode", "country", "period"])
g["EventRootCode"] = g["EventRootCode"].astype(int)
g["period"] = g["period"].astype(int)

found = set(g["country"].unique())
missing = [c for c in EXPECTED_COUNTRIES if c not in found]
print(f"  Rows: {len(g):,}")
print(f"  Countries found: {len(found & set(EXPECTED_COUNTRIES))} of {len(EXPECTED_COUNTRIES)}")
if missing:
    print(f"  WARNING - no GDELT rows for: {', '.join(missing)}")
print(f"  Event categories found: {sorted(int(c) for c in g['EventRootCode'].unique())}")

all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13)]
pivots = {}
for c, sub in g.groupby("country"):
    counts = sub.pivot_table(
        index="period", columns="EventRootCode", values="n_events", aggfunc="sum"
    )
    pivots[c] = counts.reindex(all_periods).fillna(0)


def explain(country, peak_period, window):
    """Return the event categories that spiked around peak_period for a country."""
    if country not in pivots:
        return None

    counts = pivots[country]
    totals = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.div(totals, axis=0).fillna(0)  # category share of the country's events

    idx = pd.Series([month_index(p) for p in counts.index], index=counts.index)
    peak_idx = month_index(peak_period)
    in_window = (idx >= peak_idx - window) & (idx <= peak_idx + window)

    prior = idx < (peak_idx - window)
    baseline_mask = prior if prior.sum() >= MIN_PRIOR_MONTHS else ~in_window

    rows = []
    for cat in counts.columns:
        base_count_mean = counts.loc[baseline_mask, cat].mean()
        if base_count_mean < MIN_BASELINE_COUNT:
            continue
        base_share_mean = shares.loc[baseline_mask, cat].mean()
        base_share_std = shares.loc[baseline_mask, cat].std()
        if not base_share_std or np.isnan(base_share_std):
            continue

        window_z = (shares.loc[in_window, cat] - base_share_mean) / base_share_std
        best_period = window_z.idxmax()
        z = window_z.max()
        times_typical = counts.loc[best_period, cat] / base_count_mean
        if z < MIN_SPIKE_Z or times_typical < MIN_TIMES_TYPICAL:
            continue

        rows.append({
            "event_category": CAMEO_NAMES.get(int(cat), str(cat)),
            "category_code": int(cat),
            "peak_month": period_str(best_period),
            "events_that_month": int(counts.loc[best_period, cat]),
            "times_typical": round(times_typical, 1),
            "spike_z": round(float(z), 1),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("spike_z", ascending=False).head(TOP_CATEGORIES)


# ============ LOAD TRADE ANOMALIES ============

print()
fname = input("Trade analysis file (e.g. event_analysis_2022_02.csv): ").strip()
top_n = int(input("How many top anomalous series to explain [10]: ").strip() or 10)
window = int(input("Months before/after the anomaly to search [3]: ").strip() or 3)

df = pd.read_csv(fname)
country_col = "reporterISO" if "reporterISO" in df.columns else "partnerISO"
label_col = "reporterDesc" if "reporterDesc" in df.columns else "partnerISO"

eval_df = df[~df["is_baseline_period"].astype(bool)].copy()
series_cols = list(dict.fromkeys([country_col, label_col, "commodity_name", "flowDesc"]))

ranking = (
    eval_df.groupby(series_cols)["z_score"]
    .apply(lambda x: x.abs().mean())
    .reset_index(name="avg_abs_z")
    .sort_values("avg_abs_z", ascending=False)
    .head(top_n)
)

print(f"\nExplaining the {len(ranking)} most anomalous series from {fname}")
print("(Events below are POTENTIALLY RELATED - timing overlap, not proof of cause.)\n")

output_rows = []

for _, r in ranking.iterrows():
    series = eval_df[
        (eval_df[country_col] == r[country_col])
        & (eval_df["commodity_name"] == r["commodity_name"])
        & (eval_df["flowDesc"] == r["flowDesc"])
        & (eval_df["z_anomaly"].astype(bool))
    ]
    if series.empty:
        continue

    # Report the ONSET of the anomaly (first flagged month), not the single
    # month with the biggest |z-score|. Once a series stops being reported,
    # every following month is filled as 0, and the gap between that flat 0
    # and a still-growing trend line keeps widening - so the "biggest" z-score
    # is always the LAST zero month, not when the disruption actually began.
    series_sorted = series.sort_values("period")
    peak = series_sorted.iloc[0]
    peak_period = int(peak["period"])
    direction = "below" if peak["z_score"] < 0 else "above"

    run_end = series_sorted[series_sorted["period"] >= peak_period]["period"].max()
    if run_end != peak_period:
        print(f"  (Anomaly continues from {period_str(peak_period)} through {period_str(int(run_end))} "
              f"- reporting onset shown below, not the full run)")

    print("-" * 70)
    print(f"{r[label_col]} | {r['commodity_name']} | {r['flowDesc']}   (avg |z| = {r['avg_abs_z']:.1f})")
    print(f"  Strongest anomaly: {period_str(peak_period)}  "
          f"(z = {peak['z_score']:.1f}, {direction} its expected trend, "
          f"trade value ${peak['trade_value']:,.0f})")

    result = explain(r[country_col], peak_period, window)
    if result is None:
        print(f"  No GDELT data for country code '{r[country_col]}'.")
        continue
    if result.empty:
        print("  No event category clearly stood out in this window.")
        continue

    print(f"  Event categories that spiked within +/-{window} months:")
    for i, e in enumerate(result.itertuples(), 1):
        print(f"    {i}. {e.event_category} - peak {e.peak_month}: "
              f"{e.events_that_month:,} events ({e.times_typical}x typical, z = {e.spike_z})")
        output_rows.append({
            "country": r[label_col],
            "commodity": r["commodity_name"],
            "flow": r["flowDesc"],
            "anomaly_month": period_str(peak_period),
            "anomaly_z": round(float(peak["z_score"]), 1),
            "event_category": e.event_category,
            "event_peak_month": e.peak_month,
            "events_that_month": e.events_that_month,
            "times_typical": e.times_typical,
            "event_spike_z": e.spike_z,
        })

if output_rows:
    pd.DataFrame(output_rows).to_csv("anomaly_event_links.csv", index=False)
    print("\nSaved anomaly_event_links.csv")
else:
    print("\nNo links found - try a wider window or more series.")
