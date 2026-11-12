"""
what_if_simulator.py
---------------------
VULNERABILITY / RISK MONTE CARLO SIMULATOR.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from analyze_v4 import (
    load_full_grid, fit_series_baseline, period_str, months_between,
    MONTHLY_CACHE, MIRROR_CACHE, PARTNER_ISO_TO_NAME,
)

GROUP_COLS = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]
RESPONSE_WINDOW_MONTHS = 24
N_SIMULATIONS = 1000
BLOCK_SIZE = 3

ANALOG_LIBRARY = [
    ("COVID wheat export ban (Russia)",     "Russian Federation", "Wheat",                "Export", 202003, "reporter"),
    ("COVID wheat export ban (Kazakhstan)", "Kazakhstan",         "Wheat",                "Export", 202003, "reporter"),
    ("EU potash sanctions (Belarus)",       "Belarus",            "Potassic_fertilizers", "Export", 202106, "mirror"),
    ("War sanctions - wheat (Russia)",      "Russian Federation", "Wheat",                "Export", 202202, "mirror"),
    ("War sanctions - coal (Russia)",       "Russian Federation", "Coal",                 "Export", 202208, "mirror"),
    ("War sanctions - crude oil (Russia)",  "Russian Federation", "Crude_oil",            "Export", 202212, "mirror"),
    ("War sanctions - natural gas (Russia)","Russian Federation", "Natural_gas",          "Export", 202202, "mirror"),
]

def _load_source(source):
    if source == "mirror":
        if not os.path.exists(MIRROR_CACHE):
            raise FileNotFoundError(f"Mirror cache '{MIRROR_CACHE}' not found.")
        raw = load_full_grid(MIRROR_CACHE, ["partnerISO", "cmdCode", "commodity_name", "flowDesc"])
        raw = raw[raw["partnerISO"].isin(PARTNER_ISO_TO_NAME)].copy()
        raw["reporterISO"] = raw["partnerISO"]
        raw["reporterDesc"] = raw["partnerISO"].map(PARTNER_ISO_TO_NAME)
        return raw[GROUP_COLS + ["period", "trade_value", "is_observed"]]
    if not os.path.exists(MONTHLY_CACHE):
        raise FileNotFoundError(f"Reporter-side cache '{MONTHLY_CACHE}' not found.")
    return load_full_grid(MONTHLY_CACHE, GROUP_COLS)

_SOURCE_CACHE = {}

def _get_source(source):
    if source not in _SOURCE_CACHE:
        _SOURCE_CACHE[source] = _load_source(source)
    return _SOURCE_CACHE[source]

def extract_response_profile(reporter, commodity, flow, event_start, source):
    full = _get_source(source)
    series = full[(full["reporterDesc"] == reporter)
                  & (full["commodity_name"] == commodity)
                  & (full["flowDesc"] == flow)]
    if len(series) == 0:
        return None
    out = fit_series_baseline(series, event_start, GROUP_COLS)
    if out is None:
        return None
    out = out.sort_values("period").copy()
    out["months_since_onset"] = out["period"].apply(lambda p: months_between(event_start, p))
    window = out[(out["months_since_onset"] >= 0) & (out["months_since_onset"] <= RESPONSE_WINDOW_MONTHS)].copy()
    if len(window) == 0:
        return None
    
    actual = np.expm1(window["log_value"])
    trend = np.expm1(window["trend_pred"])
    window["pct_deviation"] = (actual - trend) / trend.replace(0, np.nan) * 100
    return window[["months_since_onset", "pct_deviation"]].dropna().reset_index(drop=True)

def build_analog_library_profiles(strict=True):
    profiles, skipped = [], []
    for label, r, c, f, start, src in ANALOG_LIBRARY:
        prof = extract_response_profile(r, c, f, start, src)
        if prof is None:
            skipped.append({"analog": label, "reporter": r, "commodity": c,
                            "flow": f, "source": src, "event_start": start})
            continue
        prof = prof.copy()
        prof["analog"] = label
        prof["commodity"] = c
        profiles.append(prof)

    if not profiles:
        if strict:
            raise RuntimeError("Analog library is EMPTY. Check caches.")
        return None, skipped

    df = pd.concat(profiles, ignore_index=True)
    return df, skipped

def run_scenario(target_reporter, target_commodity, target_flow, hypothetical_start, source,
                  analog_df=None, skipped_report=None):
    warnings = []
    if skipped_report:
        warnings.append(f"{len(skipped_report)} historical analog(s) unavailable.")
    if analog_df is None:
        analog_df, skipped_report = build_analog_library_profiles()

    commodity_df = analog_df[analog_df["commodity"] == target_commodity]
    if len(commodity_df) == 0:
        raise ValueError(f"No historical analog for commodity '{target_commodity}'. Refusing to blend unrelated commodities.")
    
    if commodity_df["analog"].nunique() < 3:
        warnings.append(f"Only {commodity_df['analog'].nunique()} analog(s) used for '{target_commodity}' - treat interval as illustrative.")

    full = _get_source(source)
    target_series = full[(full["reporterDesc"] == target_reporter)
                          & (full["commodity_name"] == target_commodity)
                          & (full["flowDesc"] == target_flow)]
    if len(target_series) == 0:
        raise ValueError(f"No data for {target_reporter} / {target_commodity} / {target_flow} in '{source}'.")

    target_out = fit_series_baseline(target_series, hypothetical_start, GROUP_COLS)
    if target_out is None:
        raise ValueError(f"Not enough observed baseline history before {period_str(hypothetical_start)}.")

    target_out = target_out.sort_values("period").reset_index(drop=True)
    baseline_part = target_out[target_out["is_baseline"]]
    
    historical_max_actual = np.expm1(baseline_part["log_value"]).max()
    if pd.isna(historical_max_actual) or historical_max_actual <= 0:
        raise ValueError("Historical data is invalid or zero.")

    model = LinearRegression().fit(baseline_part[["time_idx"]].values, baseline_part["log_value"].values)
    last_idx = baseline_part["time_idx"].max()

    trend_values = np.zeros(RESPONSE_WINDOW_MONTHS + 1)
    for offset in range(0, RESPONSE_WINDOW_MONTHS + 1):
        future_time_idx = last_idx + 1 + offset
        projected_val = np.expm1(model.predict([[future_time_idx]])[0])
        trend_values[offset] = min(projected_val, historical_max_actual)

    simulated_paths = np.zeros((N_SIMULATIONS, RESPONSE_WINDOW_MONTHS + 1))
    unique_analogs = commodity_df["analog"].unique()
    
    for sim in range(N_SIMULATIONS):
        current_month = 0
        while current_month <= RESPONSE_WINDOW_MONTHS:
            sampled_analog = np.random.choice(unique_analogs)
            analog_curve = commodity_df[commodity_df["analog"] == sampled_analog].sort_values("months_since_onset")
            
            if current_month == 0:
                start_idx = 0
            else:
                max_start = max(0, len(analog_curve) - BLOCK_SIZE)
                start_idx = np.random.randint(0, max_start + 1)
            
            for b in range(BLOCK_SIZE):
                if current_month > RESPONSE_WINDOW_MONTHS:
                    break
                shock_idx = start_idx + b
                shock_pct = analog_curve.iloc[shock_idx]["pct_deviation"] if shock_idx < len(analog_curve) else 0.0
                
                simulated_paths[sim, current_month] = max(0, trend_values[current_month] * (1 + shock_pct / 100))
                current_month += 1

    p10 = np.percentile(simulated_paths, 10, axis=0)
    p50 = np.percentile(simulated_paths, 50, axis=0)
    p90 = np.percentile(simulated_paths, 90, axis=0)

    projection = pd.DataFrame({
        "months_since_hypothetical_start": range(RESPONSE_WINDOW_MONTHS + 1),
        "projected_trend_value": trend_values,
        "low_estimate_p10": p10,
        "median_estimate_p50": p50,
        "high_estimate_p90": p90,
    })

    return projection, warnings