"""
dashboard_v2.py
----------------
v2 of the trade disruption dashboard. Three tabs:

  1. Disruption Detector
  2. What-If Scenario
  3. Validated Results

See FINAL_HANDOFF_v5_whatif_fix.md for what changed vs the original.
Run with: streamlit run dashboard_v2.py
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
GDELT_FILE = "gdelt_monthly_events.csv"
CAMEO_NAMES = {
    1: "Make public statement", 2: "Appeal", 3: "Express intent to cooperate",
    4: "Consult", 5: "Engage in diplomatic cooperation", 6: "Engage in material cooperation",
    7: "Provide aid", 8: "Yield", 9: "Investigate", 10: "Demand", 11: "Disapprove",
    12: "Reject", 13: "Threaten", 14: "Protest", 15: "Exhibit force posture",
    16: "Reduce relations (sanctions, boycotts)", 17: "Coerce (restrictions, seizures)",
    18: "Assault", 19: "Fight", 20: "Use unconventional mass violence",
}
KNOWN_EVENTS = {
    "Russia-Ukraine War (Feb 2022)": "2022-02",
    "COVID-19 Pandemic / Export Restrictions (Mar 2020)": "2020-03",
    "Belarus EU Sanctions (Jun 2021)": "2021-06",
    "Suez Canal Blockage (Mar 2021)": "2021-03",
    "US-China Trade War Escalation (Jul 2018)": "2018-07",
    "Custom date...": None,
}

st.set_page_config(page_title="Trade Disruption Explorer v2", layout="wide")
st.title("Global Trade Disruption Explorer v2")
st.caption(
    "Same statistical core as the validated analyze_v4.py pipeline (imported directly, not "
    "reimplemented). Russia and Belarus are automatically routed through mirror statistics."
)


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


@st.cache_resource
def get_analog_profiles():
    return build_analog_library_profiles()


full = load_combined_grid()
if full is None:
    st.error(f"'{MONTHLY_CACHE}' not found. Run prepare_monthly_data.py first and put the CSVs next to this script.")
    st.stop()
gdelt = load_gdelt()

# ---------- SIDEBAR (module scope; not nested inside a tab) ----------
st.sidebar.header("Detector settings")
event_choice = st.sidebar.selectbox("Event", list(KNOWN_EVENTS.keys()))
if KNOWN_EVENTS[event_choice] is None:
    event_date_str = st.sidebar.text_input("Custom event date (YYYY-MM)", "2022-02")
else:
    event_date_str = KNOWN_EVENTS[event_choice]
    st.sidebar.caption(f"Date: {event_date_str}")
ey, em = map(int, event_date_str.split("-"))
event_period = ey * 100 + em

months_of_history = (ey - 2017) * 12 + (em - 1)
if months_of_history < MIN_BASELINE_MONTHS:
    st.sidebar.warning(
        f"Only {months_of_history} months of data exist before this date - series need "
        f"{MIN_BASELINE_MONTHS}+ months of baseline history for a reliable trend fit."
    )

detector = st.sidebar.radio(
    "Detector",
    ["Robust (MAD) z-score", "Standard z-score", "CUSUM (sustained change)"],
)
threshold = st.sidebar.slider("Anomaly threshold", 1.0, 5.0, 2.0, 0.25)

country_options = ["All"] + sorted(full["reporterDesc"].dropna().unique().tolist())
country_val = st.sidebar.selectbox("Country", country_options)
commodity_options = ["All"] + sorted(full["commodity_name"].dropna().unique().tolist())
commodity_val = st.sidebar.selectbox("Commodity", commodity_options)
run = st.sidebar.button("Run analysis", type="primary")

tab_detect, tab_whatif, tab_results = st.tabs(
    ["Disruption Detector", "What-If Scenario", "Validated Results"]
)

# ============================== TAB 1: DISRUPTION DETECTOR ==============================
with tab_detect:
    if run:
        filtered = full
        if country_val != "All":
            filtered = filtered[filtered["reporterDesc"] == country_val]
        if commodity_val != "All":
            filtered = filtered[filtered["commodity_name"] == commodity_val]

        gaps = reporting_gap_diagnostic(filtered)
        relevant_gaps = gaps[(gaps["period"] >= event_period - 24) & (gaps["period"] <= event_period + 24)]
        if len(relevant_gaps) > 0:
            gap_countries = relevant_gaps["reporterDesc"].unique().tolist()
            st.info(
                f"Reporting gap diagnostic: {gap_countries} have months with <5% data coverage near this "
                f"event. Those months are EXCLUDED from the trend fit, not treated as zero trade."
            )

        with st.spinner("Fitting trend + computing anomaly scores (same code as analyze_v4.py)..."):
            final = build_fold(filtered, GROUP_COLS, event_period)

        if final is None:
            st.warning("Not enough data for that combination. Try a broader filter or a different date.")
        else:
            missing = [c for c in GROUP_COLS if c not in final.columns]
            if missing:
                st.error(f"Internal error: build_fold returned a frame missing {missing}.")
            else:
                if detector == "CUSUM (sustained change)":
                    results = []
                    for _, g in final.groupby(GROUP_COLS):
                        results.append(cusum_flags(g, threshold))
                    final = pd.concat(results, ignore_index=True)
                    final["z_anomaly"] = (final["cusum_pred"] == 1) & (~final["is_baseline"])
                    rank_col = "cusum_pred"
                else:
                    z_col = "z_robust" if detector.startswith("Robust") else "z_std"
                    final["z_anomaly"] = (final[z_col].abs() > threshold) & (~final["is_baseline"])
                    rank_col = z_col

                eval_only = final[~final["is_baseline"]]
                if rank_col == "cusum_pred":
                    ranking = (
                        eval_only.groupby(GROUP_COLS)["cusum_pred"].sum()
                        .reset_index(name="score").sort_values("score", ascending=False)
                    )
                    score_label = "Months flagged (CUSUM)"
                else:
                    ranking = (
                        eval_only.groupby(GROUP_COLS)[rank_col]
                        .apply(lambda x: x.abs().mean())
                        .reset_index(name="score").sort_values("score", ascending=False)
                    )
                    score_label = "Avg |Z-score|"

                if len(ranking) == 0:
                    st.warning("No eligible series (need 24+ months of baseline history) for this combination.")
                else:
                    st.success(f"Analyzed {ranking.shape[0]} series around {event_date_str} using {detector}.")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("Top affected series")
                        display = ranking.head(15).copy()
                        display["series"] = (
                            display["reporterDesc"].astype(str) + " | "
                            + display["commodity_name"] + " | " + display["flowDesc"]
                        )
                        st.dataframe(
                            display[["series", "score"]].rename(columns={"score": score_label}),
                            hide_index=True, width="stretch",
                        )

                    with col2:
                        st.subheader("Top series over time")
                        top = ranking.iloc[0]
                        series = final[
                            (final["reporterDesc"] == top["reporterDesc"])
                            & (final["commodity_name"] == top["commodity_name"])
                            & (final["flowDesc"] == top["flowDesc"])
                        ].sort_values("period").copy()
                        series["date"] = series["period"].apply(period_to_date)
                        series["trend_value"] = np.expm1(series["trend_pred"])

                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(series["date"], series["trade_value"], color="#2563eb", linewidth=1.5, label="Actual")
                        ax.plot(series["date"], series["trend_value"], color="#94a3b8", linewidth=1.3,
                                linestyle="--", label="Expected trend")
                        anomalies = series[series["z_anomaly"] == True]
                        ax.scatter(anomalies["date"], anomalies["trade_value"], color="#dc2626", s=35,
                                   zorder=3, label="Anomaly")
                        ax.axvline(period_to_date(event_period), color="#f59e0b", linestyle=":", linewidth=2,
                                   label="Event date")
                        ax.set_title(f"{top['reporterDesc']} | {top['commodity_name']} | {top['flowDesc']}", fontsize=11)
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                        st.pyplot(fig)

                    if gdelt is not None and country_val != "All":
                        st.subheader("Potentially related GDELT events")
                        st.caption("Event categories unusually common for this country near the event date vs. its own normal rate. Timing overlap only - not proof of cause.")
                        iso_lookup = full[full["reporterDesc"] == country_val]["reporterISO"]
                        iso_code = iso_lookup.iloc[0] if len(iso_lookup) else None
                        if iso_code and iso_code in gdelt["country"].unique():
                            country_gdelt = gdelt[gdelt["country"] == iso_code].copy()
                            all_periods_gdelt = [
                                y * 100 + m for y in range(2017, 2026) for m in range(1, 13)
                                if y * 100 + m <= 202512
                            ]
                            counts = country_gdelt.pivot_table(
                                index="period", columns="EventRootCode",
                                values="n_events", aggfunc="sum",
                            )
                            counts = counts.reindex(all_periods_gdelt).fillna(0)
                            totals = counts.sum(axis=1).replace(0, np.nan)
                            shares = counts.div(totals, axis=0).fillna(0)
                            window_mask = (counts.index >= event_period - 6) & (counts.index <= event_period + 6)
                            baseline_mask = ~window_mask
                            rows = []
                            for cat in counts.columns:
                                base_count_mean = counts.loc[baseline_mask, cat].mean()
                                if base_count_mean < 20:
                                    continue
                                base_share_mean = shares.loc[baseline_mask, cat].mean()
                                base_share_std = shares.loc[baseline_mask, cat].std()
                                if not base_share_std or np.isnan(base_share_std):
                                    continue
                                window_z = (shares.loc[window_mask, cat] - base_share_mean) / base_share_std
                                if len(window_z) == 0:
                                    continue
                                best_period = window_z.idxmax()
                                z = window_z.max()
                                times_typical = counts.loc[best_period, cat] / base_count_mean
                                if z < 3.0 or times_typical < 1.5:
                                    continue
                                rows.append({
                                    "Event category": CAMEO_NAMES.get(int(cat), str(cat)),
                                    "Peak month": period_to_date(best_period).strftime("%Y-%m"),
                                    "Events that month": int(counts.loc[best_period, cat]),
                                    "x typical": round(float(times_typical), 1),
                                    "Spike z-score": round(float(z), 1),
                                })
                            if rows:
                                st.dataframe(
                                    pd.DataFrame(rows).sort_values("Spike z-score", ascending=False).head(5),
                                    hide_index=True, width="stretch",
                                )
                            else:
                                st.write("No event category clearly spiked above normal in this window.")
    else:
        st.info("Set your filters in the sidebar, then click **Run analysis**.")

# ============================== TAB 2: WHAT-IF SCENARIO ==============================
with tab_whatif:
    st.subheader("Project a hypothetical future disruption")
    st.caption(
        "NOT a trained predictor - transfers the real, observed trend-deviation curve from named "
        "historical analogs onto your chosen series' own trend. Analogs auto-narrow to the same "
        "commodity; if no same-commodity analog exists, this tab refuses to run rather than "
        "blend unrelated commodities."
    )

    try:
        profiles, skipped = get_analog_profiles()
    except FileNotFoundError as e:
        st.error(
            f"Missing cache file needed to build the analog library: {e}\n\n"
            f"Place `{MONTHLY_CACHE}` and `{MIRROR_CACHE}` next to this script and reload."
        )
        st.stop()
    except RuntimeError as e:
        st.error(f"Could not build the analog library: {e}")
        st.stop()

    st.success(
        f"Analog library loaded: {profiles['analog'].nunique()} usable analog(s) "
        f"({profiles['analog'].unique().tolist()})."
    )
    if skipped:
        with st.expander(f"{len(skipped)} historical analog(s) could not be extracted (click for details)"):
            st.dataframe(pd.DataFrame(skipped), hide_index=True, width="stretch")

    all_countries = sorted(full["reporterDesc"].dropna().unique().tolist())
    all_commodities = sorted(full["commodity_name"].dropna().unique().tolist())

    default_country = "Ukraine" if "Ukraine" in all_countries else all_countries[0]
    default_commodity = "Wheat" if "Wheat" in all_commodities else all_commodities[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        wi_country = st.selectbox(
            "Country", all_countries,
            index=all_countries.index(default_country), key="wi_country",
        )
    with c2:
        wi_commodity = st.selectbox(
            "Commodity", all_commodities,
            index=all_commodities.index(default_commodity), key="wi_commodity",
        )
    with c3:
        wi_flow = st.selectbox("Flow", ["Export", "Import"], key="wi_flow")
    with c4:
        wi_date = st.text_input("Hypothetical start (YYYY-MM)", "2024-01", key="wi_date")

    wi_source = "mirror" if wi_country in PARTNER_ISO_TO_NAME.values() else "reporter"
    st.caption(
        f"Target series source: `{wi_source}` "
        f"({'Russia/Belarus use mirror statistics automatically' if wi_source == 'mirror' else 'reporter-side data'})."
    )

    with st.expander("Historical analog library (what this projection can draw on)"):
        st.dataframe(
            pd.DataFrame(
                ANALOG_LIBRARY,
                columns=["Analog", "Reporter", "Commodity", "Flow", "Onset (YYYYMM)", "Source"],
            ),
            hide_index=True, width="stretch",
        )

    if st.button("Run What-If Projection", type="primary"):
        try:
            wy, wm = map(int, wi_date.split("-"))
            wi_start = wy * 100 + wm
            projection, analogs_used, warns = run_scenario(
                target_reporter=wi_country, target_commodity=wi_commodity, target_flow=wi_flow,
                hypothetical_start=wi_start, source=wi_source,
                analog_profiles=profiles, skipped_report=skipped,
            )
            st.markdown(f"**Analogs used:** {', '.join(analogs_used['analog'].tolist())}")
            for w in warns:
                st.warning(w)

            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(projection["months_since_hypothetical_start"], projection["median_estimate"],
                    color="#2563eb", linewidth=2, label="Median estimate")
            ax.fill_between(projection["months_since_hypothetical_start"],
                            projection["low_estimate"], projection["high_estimate"],
                            color="#2563eb", alpha=0.15, label="Range across analogs")
            ax.plot(projection["months_since_hypothetical_start"], projection["projected_trend_value"],
                    color="#94a3b8", linestyle="--", linewidth=1.3, label="Trend with no disruption")
            ax.set_xlabel("Months since hypothetical start")
            ax.set_ylabel("Projected trade value (USD)")
            ax.set_title(f"What if: {wi_country} {wi_commodity} {wi_flow.lower()}s disrupted from {wi_date}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            st.dataframe(projection, hide_index=True, width="stretch")
        except FileNotFoundError as e:
            st.error(f"Missing cache file: {e}")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error running the scenario: {type(e).__name__}: {e}")

# ============================== TAB 3: VALIDATED RESULTS ==============================
with tab_results:
    st.subheader("Validated results from analyze_v4.py (read-only)")
    st.caption("These are NOT recomputed live here - they're the saved output of the validated pipeline, shown for reference during a demo.")
    if os.path.exists("v4_summary.txt"):
        with open("v4_summary.txt") as f:
            st.text(f.read())
    else:
        st.info("v4_summary.txt not found in this folder - run analyze_v4.py first to generate it.")

    for fname, label in [
        ("v4_test_fold_comparison.csv", "Test-fold detector comparison"),
        ("v4_onset_recall.csv", "Onset recall (caught within 6mo of real start)"),
        ("v4_early_warning_results.csv", "Early warning (walk-forward, no lookahead)"),
    ]:
        if os.path.exists(fname):
            st.markdown(f"**{label}**")
            st.dataframe(pd.read_csv(fname), hide_index=True, width="stretch")