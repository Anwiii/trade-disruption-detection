"""
plot_summary.py
----------------
Builds the full visualization set for the report/demo:

1. Ranked bar charts - top affected country-commodity-flow pairs
   (for both the war and COVID event analyses)
2. Early-warning chart - Belarus energy exports, showing the lead
   window (pre-official-event) separately from the main disruption
3. GDELT correlation chart - trade anomaly (z-score) over time next
   to the GDELT event spike over time, for the same country/window

Run this AFTER you have:
  - event_analysis_2022_02.csv and event_analysis_2020_03.csv (from analyze_event.py)
  - mirror_disruption_results.csv (from analyze_mirror_disruption.py)
  - early_warning_results.csv + the mirror cache (from test_early_warning.py)
  - gdelt_monthly_events.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.linear_model import LinearRegression

CAMEO_NAMES = {
    1: "Make public statement", 2: "Appeal", 3: "Express intent to cooperate",
    4: "Consult", 5: "Engage in diplomatic cooperation",
    6: "Engage in material cooperation", 7: "Provide aid", 8: "Yield",
    9: "Investigate", 10: "Demand", 11: "Disapprove", 12: "Reject",
    13: "Threaten", 14: "Protest", 15: "Exhibit force posture",
    16: "Reduce relations (sanctions, boycotts)",
    17: "Coerce (restrictions, seizures)", 18: "Assault", 19: "Fight",
    20: "Use unconventional mass violence",
}


def period_to_date(period):
    return datetime(int(period) // 100, int(period) % 100, 1)


# ============ CHART TYPE 1: RANKED BAR CHART ============

def make_ranking_chart(csv_path, label_col, title, output_name, top_n=10):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  SKIPPED ({csv_path} not found)")
        return

    eval_df = df[~df["is_baseline_period"].astype(bool)].copy()
    group_cols = [label_col, "commodity_name", "flowDesc"]
    ranking = (
        eval_df.groupby(group_cols)["z_score"]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="avg_abs_z")
        .sort_values("avg_abs_z", ascending=False)
        .head(top_n)
    )
    ranking["label"] = ranking[label_col] + " | " + ranking["commodity_name"] + " | " + ranking["flowDesc"]
    ranking = ranking.sort_values("avg_abs_z")  # ascending for horizontal bar (largest at top)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(ranking["label"], ranking["avg_abs_z"], color="#2563eb")
    ax.set_xlabel("Average |Z-score| during evaluation period")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"  Saved: {output_name}")


# ============ CHART TYPE 2: EARLY WARNING CHART ============

def make_early_warning_chart(mirror_cache, partner_iso, commodity, flow,
                              official_event_period, lead_start_period, output_name):
    try:
        pdf = pd.read_csv(mirror_cache)
    except FileNotFoundError:
        print(f"  SKIPPED ({mirror_cache} not found)")
        return

    pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]
    series = pdf[
        (pdf["partnerISO"] == partner_iso)
        & (pdf["commodity_name"] == commodity)
        & (pdf["flowDesc"] == flow)
    ].copy()

    if len(series) == 0:
        print(f"  SKIPPED: no data for {partner_iso}/{commodity}/{flow}")
        return

    all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13) if y * 100 + m <= 202512]
    full = pd.DataFrame({"period": all_periods}).merge(series[["period", "trade_value"]], on="period", how="left")
    full["trade_value"] = full["trade_value"].fillna(0)
    full = full.sort_values("period").reset_index(drop=True)
    full["time_idx"] = range(len(full))
    full["date"] = full["period"].apply(period_to_date)

    # Fit trend using only data before the lead window starts (true "no foreknowledge" baseline)
    baseline = full[full["period"] < lead_start_period].copy()
    baseline["log_value"] = np.log1p(baseline["trade_value"])
    full["log_value"] = np.log1p(full["trade_value"])

    model = LinearRegression()
    model.fit(baseline[["time_idx"]].values, baseline["log_value"].values)
    full["trend_pred"] = model.predict(full[["time_idx"]].values)
    full["trend_value"] = np.expm1(full["trend_pred"])

    lead_date = period_to_date(lead_start_period)
    event_date = period_to_date(official_event_period)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(full["date"], full["trade_value"], color="#2563eb", linewidth=1.5, label="Actual trade value (via mirror statistics)")
    ax.plot(full["date"], full["trend_value"], color="#94a3b8", linewidth=1.5, linestyle="--", label="Expected trend (pre-lead-window baseline)")

    ax.axvspan(lead_date, event_date, color="#fbbf24", alpha=0.2, label="Lead window (pre-official-event)")
    ax.axvline(event_date, color="#dc2626", linestyle=":", linewidth=2, label="Official event date")

    ax.set_title(f"{partner_iso} {commodity} {flow} - Early Warning Window\n(via mirror statistics)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trade Value (USD)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Zoom to a reasonable window around the event for readability
    ax.set_xlim(period_to_date(lead_start_period - 300), period_to_date(official_event_period + 600))

    plt.tight_layout()
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"  Saved: {output_name}")


# ============ CHART TYPE 3: GDELT CORRELATION CHART ============

def make_gdelt_correlation_chart(trade_csv, gdelt_csv, country_col, country_val,
                                  commodity, flow, event_category_code, output_name,
                                  window_start_period, window_end_period,
                                  gdelt_country_code=None):
    # GDELT's own 'country' column uses ISO3 codes (e.g. "BLR"), not full
    # names like "Belarus" - gdelt_country_code lets you pass that separately
    # from country_val, which matches the trade CSV's naming instead.
    if gdelt_country_code is None:
        gdelt_country_code = country_val
    try:
        trade = pd.read_csv(trade_csv)
        gdelt = pd.read_csv(gdelt_csv)
    except FileNotFoundError as e:
        print(f"  SKIPPED ({e.filename} not found)")
        return

    series = trade[
        (trade[country_col] == country_val)
        & (trade["commodity_name"] == commodity)
        & (trade["flowDesc"] == flow)
    ].copy()
    if len(series) == 0:
        print(f"  SKIPPED: no trade data for {country_val}/{commodity}/{flow}")
        return
    series = series.sort_values("period")
    series = series[(series["period"] >= window_start_period) & (series["period"] <= window_end_period)]
    series["date"] = series["period"].apply(period_to_date)

    gdelt_country = gdelt[gdelt["country"] == gdelt_country_code].copy()
    gdelt_country["EventRootCode"] = pd.to_numeric(gdelt_country["EventRootCode"], errors="coerce")
    cat = gdelt_country[gdelt_country["EventRootCode"] == event_category_code].copy()
    cat = cat[(cat["period"] >= window_start_period) & (cat["period"] <= window_end_period)]
    cat["date"] = cat["period"].apply(period_to_date)
    cat = cat.sort_values("period")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(series["date"], series["z_score"], color="#2563eb", linewidth=1.8, marker="o", markersize=4)
    ax1.axhline(0, color="#94a3b8", linewidth=1)
    ax1.axhline(2.5, color="#dc2626", linewidth=1, linestyle="--", alpha=0.6)
    ax1.axhline(-2.5, color="#dc2626", linewidth=1, linestyle="--", alpha=0.6)
    ax1.set_ylabel("Trade Z-score")
    ax1.set_title(f"{country_val} {commodity} {flow}: Trade Anomaly vs. GDELT '{CAMEO_NAMES.get(event_category_code, event_category_code)}' Events",
                  fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2.bar(cat["date"], cat["n_events"], width=20, color="#dc2626", alpha=0.7)
    ax2.set_ylabel(f"'{CAMEO_NAMES.get(event_category_code, event_category_code)}'\nevent count")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"  Saved: {output_name}")


# ============ RUN ALL CHARTS ============

print("1. Ranking charts")
make_ranking_chart("event_analysis_2022_02.csv", "reporterDesc",
                    "Top Affected Series: Russia-Ukraine War (Feb 2022)",
                    "chart_ranking_war.png")
make_ranking_chart("event_analysis_2020_03.csv", "reporterDesc",
                    "Top Affected Series: COVID-19 Disruption (Mar 2020)",
                    "chart_ranking_covid.png")

print("\n2. Early warning chart (Belarus energy exports)")
make_early_warning_chart("mirror_monthly_series_cache.csv", "BLR", "Crude_oil", "Export",
                          official_event_period=202202, lead_start_period=202108,
                          output_name="chart_early_warning_belarus_oil.png")
make_early_warning_chart("mirror_monthly_series_cache.csv", "BLR", "Natural_gas", "Export",
                          official_event_period=202202, lead_start_period=202108,
                          output_name="chart_early_warning_belarus_gas.png")

print("\n3. GDELT correlation chart (Belarus, 'Reduce relations' category)")
make_gdelt_correlation_chart("event_analysis_2022_02.csv", "gdelt_monthly_events.csv",
                              "reporterDesc", "Belarus", "Sunflower_oil", "Import",
                              event_category_code=16,
                              output_name="chart_gdelt_correlation_belarus.png",
                              window_start_period=202101, window_end_period=202212,
                              gdelt_country_code="BLR")

print("\nDone. Check your BDA folder for the chart_*.png files.")
