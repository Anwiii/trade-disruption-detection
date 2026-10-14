"""
dashboard.py
------------
Interactive web dashboard for the trade disruption project.
Pick a known event (or a custom date), optionally filter by country/
commodity, and see the results live: a ranked table, a chart of the
top-affected series, and - if available - matching GDELT events.

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.linear_model import LinearRegression
import os

MONTHLY_CACHE = "monthly_series_cache.csv"
MIRROR_CACHE = "mirror_monthly_series_cache.csv"
GDELT_FILE = "gdelt_monthly_events.csv"
Z_SCORE_THRESHOLD = 2.5

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

KNOWN_EVENTS = {
    "Russia-Ukraine War (Feb 2022)": "2022-02",
    "COVID-19 Pandemic / Export Restrictions (Mar 2020)": "2020-03",
    "Belarus EU Sanctions (Jun 2021)": "2021-06",
    "Suez Canal Blockage (Mar 2021)": "2021-03",
    "US-China Trade War Escalation (Jul 2018)": "2018-07",
    "Custom date...": None,
}

st.set_page_config(page_title="Trade Disruption Explorer", layout="wide")
st.title("Global Trade Disruption Explorer")
st.caption("Detects unusual trade changes around a chosen event, using a trend-adjusted baseline over UN Comtrade data (2017-2025).")


def period_to_date(period):
    return datetime(int(period) // 100, int(period) % 100, 1)


@st.cache_data
def load_monthly():
    if not os.path.exists(MONTHLY_CACHE):
        return None
    df = pd.read_csv(MONTHLY_CACHE)
    df["period"] = df["refYear"] * 100 + df["refMonth"]
    return df


@st.cache_data
def load_mirror():
    if not os.path.exists(MIRROR_CACHE):
        return None
    df = pd.read_csv(MIRROR_CACHE)
    df["period"] = df["refYear"] * 100 + df["refMonth"]
    return df


@st.cache_data
def load_gdelt():
    if not os.path.exists(GDELT_FILE):
        return None
    g = pd.read_csv(GDELT_FILE)
    g["EventRootCode"] = pd.to_numeric(g["EventRootCode"], errors="coerce")
    g = g.dropna(subset=["EventRootCode", "country", "period"])
    g["EventRootCode"] = g["EventRootCode"].astype(int)
    g["period"] = g["period"].astype(int)
    return g


def run_analysis(pdf, event_period, group_cols, country_filter_col, country_val, commodity_val):
    data = pdf.copy()
    if country_val != "All" and country_filter_col in data.columns:
        data = data[data[country_filter_col] == country_val]
    if commodity_val != "All":
        data = data[data["commodity_name"] == commodity_val]

    if len(data) == 0:
        return None, None

    all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13) if y * 100 + m <= 202512]
    series_keys = data[group_cols].drop_duplicates()
    full_index = series_keys.merge(pd.DataFrame({"period": all_periods}), how="cross")
    full = full_index.merge(data[group_cols + ["period", "trade_value"]], on=group_cols + ["period"], how="left")
    full["trade_value"] = full["trade_value"].fillna(0)

    results = []
    for _, group_df in full.groupby(group_cols):
        group_df = group_df.sort_values("period").reset_index(drop=True)
        group_df["time_idx"] = range(len(group_df))

        baseline = group_df[group_df["period"] < event_period].copy()
        evaluation = group_df[group_df["period"] >= event_period]

        if len(baseline) < 12 or (baseline["trade_value"] > 0).sum() < 6 or len(evaluation) < 1:
            continue

        baseline["log_value"] = np.log1p(baseline["trade_value"])
        group_df["log_value"] = np.log1p(group_df["trade_value"])

        model = LinearRegression()
        model.fit(baseline[["time_idx"]].values, baseline["log_value"].values)
        group_df["trend_pred"] = model.predict(group_df[["time_idx"]].values)
        group_df["residual"] = group_df["log_value"] - group_df["trend_pred"]

        std = group_df.loc[group_df["period"] < event_period, "residual"].std()
        if std == 0 or np.isnan(std):
            continue

        group_df["z_score"] = group_df["residual"] / std
        group_df["is_baseline"] = group_df["period"] < event_period
        group_df["z_anomaly"] = (group_df["z_score"].abs() > Z_SCORE_THRESHOLD) & (~group_df["is_baseline"])
        results.append(group_df)

    if not results:
        return None, None

    final = pd.concat(results, ignore_index=True)
    eval_only = final[~final["is_baseline"]]
    ranking = (
        eval_only.groupby(group_cols)["z_score"]
        .apply(lambda x: x.abs().mean())
        .reset_index(name="avg_abs_z")
        .sort_values("avg_abs_z", ascending=False)
    )
    return final, ranking


# ============ SIDEBAR CONTROLS ============

monthly = load_monthly()
mirror = load_mirror()
gdelt = load_gdelt()

if monthly is None:
    st.error(f"'{MONTHLY_CACHE}' not found. Run prepare_monthly_data.py first.")
    st.stop()

st.sidebar.header("Settings")

event_choice = st.sidebar.selectbox("Event", list(KNOWN_EVENTS.keys()))
if KNOWN_EVENTS[event_choice] is None:
    event_date_str = st.sidebar.text_input("Custom event date (YYYY-MM)", "2022-02")
else:
    event_date_str = KNOWN_EVENTS[event_choice]
    st.sidebar.caption(f"Date: {event_date_str}")

ey, em = map(int, event_date_str.split("-"))
event_period = ey * 100 + em

source = st.sidebar.radio("Data source", ["UN Comtrade (reporter-side)", "Mirror statistics (Russia/Belarus via partners)"])

if source.startswith("Mirror"):
    if mirror is None:
        st.sidebar.error(f"'{MIRROR_CACHE}' not found.")
        st.stop()
    pdf = mirror
    group_cols = ["partnerISO", "cmdCode", "commodity_name", "flowDesc"]
    country_col = "partnerISO"
    country_options = ["All"] + sorted(pdf[country_col].unique().tolist())
else:
    pdf = monthly
    group_cols = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]
    country_col = "reporterDesc"
    country_options = ["All"] + sorted(pdf[country_col].unique().tolist())

country_val = st.sidebar.selectbox("Country", country_options)
commodity_options = ["All"] + sorted(pdf["commodity_name"].unique().tolist())
commodity_val = st.sidebar.selectbox("Commodity", commodity_options)

run = st.sidebar.button("Run analysis", type="primary")

# ============ MAIN PANEL ============

if run:
    with st.spinner("Analyzing..."):
        final, ranking = run_analysis(pdf, event_period, group_cols, country_col, country_val, commodity_val)

    if final is None:
        st.warning("Not enough data for that combination. Try a broader filter or a different date.")
        st.stop()

    st.success(f"Analyzed {ranking.shape[0]} series around {event_date_str}.")

    # ---- Plain-language verdict ----
    top_row = ranking.iloc[0]
    top_series_df = final[
        (final[country_col] == top_row[country_col])
        & (final["commodity_name"] == top_row["commodity_name"])
        & (final["flowDesc"] == top_row["flowDesc"])
    ].sort_values("period")

    eval_rows = top_series_df[~top_series_df["is_baseline"]]
    anomaly_rows = eval_rows[eval_rows["z_anomaly"] == True]

    if len(anomaly_rows) > 0:
        onset = anomaly_rows.iloc[0]
        onset_date = period_to_date(int(onset["period"])).strftime("%B %Y")
        actual = onset["trade_value"]
        expected = np.expm1(onset["trend_pred"])
        if expected > 0:
            pct_change = (actual - expected) / expected * 100
        else:
            pct_change = None
        direction = "below" if onset["z_score"] < 0 else "above"
        n_anomalous_months = len(anomaly_rows)
        still_ongoing = anomaly_rows.iloc[-1]["period"] == eval_rows.iloc[-1]["period"]

        verdict = (
            f"**Disruption detected: {top_row[country_col]}'s {top_row['commodity_name']} "
            f"{top_row['flowDesc'].lower()} first broke from its expected trend in {onset_date}**, "
        )
        if pct_change is not None:
            verdict += f"trading {abs(pct_change):,.0f}% {direction} expectation that month. "
        else:
            verdict += f"moving sharply {direction} its expected trend. "
        verdict += f"{n_anomalous_months} month(s) in the evaluation period were flagged as anomalous"
        verdict += ", and the disruption appears to still be ongoing as of the most recent data." if still_ongoing else "."

        st.markdown(f"> {verdict}")
    else:
        st.markdown(
            f"> **No significant disruption detected** for {top_row[country_col]}'s "
            f"{top_row['commodity_name']} {top_row['flowDesc'].lower()} (the top-ranked series) "
            f"around {event_date_str} — its avg |z-score| of {top_row['avg_abs_z']:.1f} did not "
            f"cross the anomaly threshold in enough months to register as a clear disruption."
        )

    st.caption(
        "This verdict covers only the single most-affected series (top row of the table below). "
        "Other rows may show their own, separate disruption patterns - check the chart for any series by re-running with a country/commodity filter."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Top affected series")
        label_col = country_col
        display = ranking.head(15).copy()
        display["series"] = display[label_col].astype(str) + " | " + display["commodity_name"] + " | " + display["flowDesc"]
        st.dataframe(display[["series", "avg_abs_z"]].rename(columns={"avg_abs_z": "Avg |Z-score|"}),
                     hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Top series over time")
        top = ranking.iloc[0]
        series = final[
            (final[country_col] == top[country_col])
            & (final["commodity_name"] == top["commodity_name"])
            & (final["flowDesc"] == top["flowDesc"])
        ].sort_values("period").copy()
        series["date"] = series["period"].apply(period_to_date)
        series["trend_value"] = np.expm1(series["trend_pred"])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(series["date"], series["trade_value"], color="#2563eb", linewidth=1.5, label="Actual")
        ax.plot(series["date"], series["trend_value"], color="#94a3b8", linewidth=1.3, linestyle="--", label="Expected trend")
        anomalies = series[series["z_anomaly"] == True]
        ax.scatter(anomalies["date"], anomalies["trade_value"], color="#dc2626", s=35, zorder=3, label="Anomaly")
        ax.axvline(period_to_date(event_period), color="#f59e0b", linestyle=":", linewidth=2, label="Event date")
        ax.set_title(f"{top[country_col]} | {top['commodity_name']} | {top['flowDesc']}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        st.pyplot(fig)

    # ---- GDELT panel (only for reporter-side country selections with a GDELT code) ----
    if gdelt is not None and source.startswith("UN Comtrade") and country_val != "All":
        st.subheader("Potentially related GDELT events")
        st.caption("Timing overlap only - not proof of cause.")

        iso_lookup = monthly[monthly["reporterDesc"] == country_val]["reporterISO"]
        iso_code = iso_lookup.iloc[0] if len(iso_lookup) else None

        if iso_code and iso_code in gdelt["country"].unique():
            window = gdelt[
                (gdelt["country"] == iso_code)
                & (gdelt["period"] >= event_period - 6)
                & (gdelt["period"] <= event_period + 6)
            ].copy()
            cat_totals = window.groupby("EventRootCode")["n_events"].sum().sort_values(ascending=False).head(5)
            if len(cat_totals):
                cat_table = pd.DataFrame({
                    "Event category": [CAMEO_NAMES.get(c, c) for c in cat_totals.index],
                    "Events (+/-6mo window)": cat_totals.values,
                })
                st.dataframe(cat_table, hide_index=True, use_container_width=True)
            else:
                st.write("No GDELT data for this country/window.")
        else:
            st.write(f"No GDELT data found for country code matching '{country_val}'.")
else:
    st.info("Set your filters in the sidebar, then click **Run analysis**.")
