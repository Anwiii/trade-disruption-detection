"""
dashboard_v2.py
----------------
Final Presentation Version with Policy Brief and Date Picker.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from analyze_v4 import (
    MONTHLY_CACHE, MIRROR_CACHE, PARTNER_ISO_TO_NAME, MIN_BASELINE_MONTHS,
    load_full_grid, reporting_gap_diagnostic, build_fold, cusum_flags, period_str,
)
from what_if_simulator import (
    run_scenario, build_analog_library_profiles, ANALOG_LIBRARY,
)

GROUP_COLS = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]
KNOWN_EVENTS = {
    "Russia-Ukraine War (Feb 2022)": "2022-02",
    "COVID-19 Pandemic (Mar 2020)": "2020-03",
    "Belarus EU Sanctions (Jun 2021)": "2021-06",
    "Suez Canal Blockage (Mar 2021)": "2021-03",
}

st.set_page_config(page_title="Trade Disruption Explorer", layout="wide")
st.title("Global Trade Disruption Explorer")

def period_to_date(period):
    return datetime(int(period) // 100, int(period) % 100, 1)

@st.cache_data
def load_combined_grid():
    if not os.path.exists(MONTHLY_CACHE):
        return None
    reporter_full = load_full_grid(MONTHLY_CACHE, GROUP_COLS)
    combined = reporter_full[~reporter_full["reporterDesc"].isin(PARTNER_ISO_TO_NAME.values())].copy()

    if os.path.exists(MIRROR_CACHE):
        mirror_raw = pd.read_csv(MIRROR_CACHE)
        mirror_raw["period"] = mirror_raw["refYear"] * 100 + mirror_raw["refMonth"]
        mirror_raw = mirror_raw[mirror_raw["partnerISO"].isin(PARTNER_ISO_TO_NAME.keys())].copy()
        mirror_raw["reporterISO"] = mirror_raw["partnerISO"]
        mirror_raw["reporterDesc"] = mirror_raw["partnerISO"].map(PARTNER_ISO_TO_NAME)

        all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13) if y * 100 + m <= 202512]
        mkeys = mirror_raw[GROUP_COLS].drop_duplicates()
        mfull_idx = mkeys.merge(pd.DataFrame({"period": all_periods}), how="cross")
        mfull = mfull_idx.merge(mirror_raw[GROUP_COLS + ["period", "trade_value"]], on=GROUP_COLS + ["period"], how="left")
        mfull["is_observed"] = mfull["trade_value"].notna()
        combined = pd.concat([combined, mfull], ignore_index=True)

    return combined

@st.cache_resource
def get_analog_library():
    return build_analog_library_profiles()

full = load_combined_grid()
if full is None:
    st.error(f"'{MONTHLY_CACHE}' not found.")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("Analysis Settings")
event_choice = st.sidebar.selectbox("Event to analyze", list(KNOWN_EVENTS.keys()))
event_date_str = KNOWN_EVENTS[event_choice]
ey, em = map(int, event_date_str.split("-"))
event_period = ey * 100 + em

country_options = ["All"] + sorted(full["reporterDesc"].dropna().unique().tolist())
country_val = st.sidebar.selectbox("Focus Country", country_options)
run = st.sidebar.button("Analyze Disruption", type="primary")

tab_detect, tab_whatif = st.tabs(["Detector (Explained)", "Vulnerability Simulator"])

# ============================== TAB 1: DISRUPTION DETECTOR ==============================
with tab_detect:
    if run:
        if country_val == "All":
            focus_countries = ["Russian Federation", "Belarus", "Ukraine", "Germany", "China", "Saudi Arabia", "United States", "Kazakhstan", "Argentina"]
            filtered = full[full["reporterDesc"].isin(focus_countries)]
        else:
            filtered = full[full["reporterDesc"] == country_val]
        
        with st.spinner("Calculating anomalies..."):
            final = build_fold(filtered, GROUP_COLS, event_period)

        if final is not None:
            final["is_anomaly"] = (final["z_robust"].abs() > 2.0) & (~final["is_baseline"])
            eval_only = final[~final["is_baseline"]].copy()
            
            eval_only["actual_usd"] = np.expm1(eval_only["log_value"])
            eval_only["expected_usd"] = np.expm1(eval_only["trend_pred"])
            eval_only["diff_usd"] = eval_only["actual_usd"] - eval_only["expected_usd"]
            eval_only["pct_dev"] = (eval_only["diff_usd"] / eval_only["expected_usd"].replace(0, np.nan)) * 100
            
            drops = eval_only[(eval_only["is_anomaly"] == True) & (eval_only["diff_usd"] < 0)]
            
            if len(drops) > 0:
                worst_drop_row = drops.loc[drops["pct_dev"].idxmin()]
                top_series = final[(final["reporterDesc"] == worst_drop_row["reporterDesc"]) & 
                                   (final["commodity_name"] == worst_drop_row["commodity_name"]) & 
                                   (final["flowDesc"] == worst_drop_row["flowDesc"])].sort_values("period")
                
                max_actual_date = top_series[top_series["is_observed"]]["period"].max()
                top_series.loc[top_series["period"] > max_actual_date, "trend_pred"] = np.nan

                window_start = event_period
                window_end = (ey + 1) * 100 + em
                if window_end > max_actual_date:
                    window_end = max_actual_date 
                
                window_data = top_series[(top_series["period"] >= window_start) & (top_series["period"] <= window_end)]
                window_data = window_data[window_data["is_anomaly"] == True]
                
                if len(window_data) > 0:
                    actual_window = np.expm1(window_data["log_value"]).sum()
                    expected_window = np.expm1(window_data["trend_pred"]).sum()
                    deficit = expected_window - actual_window
                    impact_text = f"${abs(deficit)/1e9:,.2f} Billion in {'lost' if deficit > 0 else 'unexpected'} trade volume."
                    direction = "collapsed" if deficit > 0 else "spiked"
                    pct_change = (deficit / expected_window) * 100 if expected_window > 0 else 0
                else:
                    impact_text = "insufficient data"
                    pct_change = 0
                    direction = "changed"

                st.markdown("---")
                st.subheader("🚨 What Happened?")
                st.markdown(f"""
                Based on the 12-month post-event window (**{period_str(window_start)}** to **{period_str(window_end)}**), 
                the most severe disruption occurred in **{worst_drop_row['reporterDesc']}'s {worst_drop_row['commodity_name']} {worst_drop_row['flowDesc']}s**.
                
                Instead of following its expected historical trend, trade volumes {direction} by **{abs(pct_change):.1f}%**, 
                resulting in an estimated {impact_text}
                """)
                st.markdown("---")

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("Actual vs. Expected Trade")
                    series_plot = top_series.copy()
                    series_plot["date"] = series_plot["period"].apply(period_to_date)
                    actual_b = np.expm1(series_plot["log_value"]) / 1e9
                    expected_b = np.expm1(series_plot["trend_pred"]) / 1e9
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(series_plot["date"], actual_b, color="#2563eb", linewidth=2, label="Actual Trade")
                    ax.plot(series_plot["date"], expected_b, color="#94a3b8", linewidth=2, linestyle="--", label="Expected Trend")
                    ax.fill_between(series_plot["date"], actual_b, expected_b, where=(actual_b < expected_b), color='red', alpha=0.2, label="Trade Deficit")
                    ax.axvline(period_to_date(event_period), color="black", linestyle=":", label="Event Date")
                    ax.set_ylabel("Trade Volume (Billions USD)")
                    ax.legend()
                    st.pyplot(fig)
                    
                with col2:
                    st.subheader("Monthly Loss/Gain (Billions USD)")
                    series_plot["diff_b"] = actual_b - expected_b
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    colors = ['red' if x < 0 else 'green' for x in series_plot["diff_b"]]
                    ax2.bar(series_plot["date"], series_plot["diff_b"], color=colors, width=20)
                    ax2.axhline(0, color="black", linewidth=1)
                    ax2.set_ylabel("Difference from Trend (Billions USD)")
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                    st.pyplot(fig2)

# ============================== TAB 2: VULNERABILITY SIMULATOR ==============================
with tab_whatif:
    st.subheader("Vulnerability Simulator (Monte Carlo Risk Assessment)")
    
    try:
        analog_df, skipped = get_analog_library()
    except Exception as e:
        st.error(f"Library error: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        wi_country = st.selectbox("Country to analyze", sorted(full["reporterDesc"].dropna().unique().tolist()), index=0)
    with c2:
        wi_commodity = st.selectbox("Commodity", sorted(full["commodity_name"].dropna().unique().tolist()), index=0)
    with c3:
        wi_date = st.text_input("Hypothetical Disruption Date (YYYY-MM)", "2025-01")

    st.info(f"""
    **Scenario Stress Test:** 
    Imagine a sudden geopolitical shock hits **{wi_country}'s {wi_commodity} exports** starting in **{wi_date}**. 
    The model will simulate 1,000 possible futures based on how past disruptions played out.
    """)

    if st.button("Run Risk Assessment", type="primary"):
        try:
            wy, wm = map(int, wi_date.split("-"))
            future_start = wy * 100 + wm
            
            projection, warns = run_scenario(
                target_reporter=wi_country, target_commodity=wi_commodity, target_flow="Export",
                hypothetical_start=future_start, 
                source="mirror" if wi_country in PARTNER_ISO_TO_NAME.values() else "reporter",
                analog_df=analog_df 
            )
            
            for w in warns: st.warning(w)
            
            # ================= GENERATE THE POLICY BRIEF =================
            baseline_avg = projection["projected_trend_value"].mean() / 1e9
            median_avg = projection["median_estimate_p50"].mean() / 1e9
            p10_avg = projection["low_estimate_p10"].mean() / 1e9
            
            loss_pct = ((median_avg - baseline_avg) / baseline_avg) * 100
            worst_case_loss = baseline_avg - p10_avg

            st.markdown("---")
            st.subheader("📋 Policy Brief: Risk Quantification")
            st.markdown(f"""
            If **{wi_country}** experiences a direct disruption to its **{wi_commodity}** exports starting in **{wi_date}**, 
            our Monte Carlo simulation (1,000 runs) projects the following 24-month impact:

            *   **Expected Baseline:** ${baseline_avg:,.2f} Billion per month (if no disruption occurs).
            *   **Median Projection:** ${median_avg:,.2f} Billion per month ({loss_pct:.1f}% change).
            *   **Worst-Case Scenario (P10):** The most severe 10% of simulations project exports dropping to **${p10_avg:,.2f} Billion per month**. This represents an estimated **${worst_case_loss:,.2f} Billion in lost trade volume per month**.
            """)
            st.markdown("---")
            # =============================================================

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(projection["months_since_hypothetical_start"], projection["median_estimate_p50"]/1e9, 
                    color="#2563eb", linewidth=2.5, label="Median Risk Path (P50)")
            
            ax.fill_between(projection["months_since_hypothetical_start"], 
                            projection["low_estimate_p10"]/1e9, 
                            projection["high_estimate_p90"]/1e9, 
                            color="#2563eb", alpha=0.2, label="80% Simulation Band (P10 - P90)")
            
            ax.plot(projection["months_since_hypothetical_start"], projection["projected_trend_value"]/1e9, 
                    color="#94a3b8", linestyle="--", linewidth=2, label="Expected Trade (No Disruption)")
            
            ax.set_xlabel("Months Since Hypothetical Disruption")
            ax.set_ylabel("Trade Volume (Billions USD)")
            ax.set_title(f"Risk Analysis: Hypothetical Disruption to {wi_country} {wi_commodity} Exports")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")