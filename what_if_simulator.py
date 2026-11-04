"""
what_if_simulator.py
---------------------
Analog-based "what if" scenario explorer: given a HYPOTHETICAL future
disruption (country, commodity, flow, start month), project a plausible
range of trend deviation.

THIS IS NOT A TRAINED PREDICTOR. There are only ~10 independently-documented
historical disruptions in this project - nowhere near enough to fit a model
that generalizes "event features -> response shape" across arbitrary future
events. Instead this tool:

  1. Treats those ~10 historical events as an analog library. For each, it
     extracts the ACTUAL observed trend-deviation curve (using the exact
     same trend-fitting code as analyze_v4.py - no separate/divergent math).
  2. Fits the TARGET series' own trend up to the hypothetical start date and
     extrapolates it forward.
  3. Overlays the analog(s)' response curve(s) onto that extrapolated trend,
     producing a min/median/max range - not a single invented number - and
     always names which real historical case(s) it came from.

This makes every projection auditable back to a real, dated event instead of
presenting a black-box forecast.

CHANGES IN THIS REVISION (see FINAL_HANDOFF_v5_whatif_fix.md):
  - build_analog_library_profiles() raises RuntimeError on an empty library
    instead of silently returning an empty frame, and reports which analog
    labels were skipped and why.
  - run_scenario() raises ValueError (instead of silently falling back to
    the full library) when no same-commodity analog exists. Blending an
    unrelated commodity into a projection is a correctness bug, not a
    "rough illustration".
  - _load_source() raises FileNotFoundError with the offending cache path
    instead of failing deep inside load_full_grid.

Usage:
    python3 what_if_simulator.py
    (edit the SCENARIO block at the bottom, or import run_scenario()
    programmatically - this is what the dashboard's "what if" tab calls)
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
RESPONSE_WINDOW_MONTHS = 24  # how many months after onset to extract / project forward

# The analog library: the SAME independently-dated, documented events already
# used as ground truth in analyze_v4.py (see VALIDATION_LABELS / TEST_LABELS
# there). Reusing them - rather than deriving new ones from this tool's own
# output - is what keeps this non-circular, same principle as the v4 fix.
ANALOG_LIBRARY = [
    # (label,                              reporter,               commodity,                 flow,     event_start, source)
    ("COVID wheat export ban (Russia)",     "Russian Federation",   "Wheat",                   "Export", 202003,      "reporter"),
    ("COVID wheat export ban (Kazakhstan)", "Kazakhstan",            "Wheat",                   "Export", 202003,      "reporter"),
    ("EU potash sanctions (Belarus)",       "Belarus",               "Potassic_fertilizers",    "Export", 202106,      "mirror"),
    ("War sanctions - wheat (Russia)",      "Russian Federation",   "Wheat",                   "Export", 202202,      "mirror"),
    ("War sanctions - coal (Russia)",       "Russian Federation",   "Coal",                     "Export", 202208,      "mirror"),
    ("War sanctions - crude oil (Russia)",  "Russian Federation",   "Crude_oil",                "Export", 202212,      "mirror"),
    ("War sanctions - natural gas (Russia)","Russian Federation",   "Natural_gas",               "Export", 202202,      "mirror"),
]


def _load_source(source):
    """Returns a full (series x period) grid with group_cols matching
    GROUP_COLS regardless of whether it came from the reporter-side or
    mirror cache - same logic as analyze_v4.main(). Raises FileNotFoundError
    with the offending path if the cache is missing."""
    if source == "mirror":
        if not os.path.exists(MIRROR_CACHE):
            raise FileNotFoundError(
                f"Mirror cache '{MIRROR_CACHE}' not found next to what_if_simulator.py. "
                f"Run prepare_monthly_data.py first, or check the path."
            )
        raw = load_full_grid(MIRROR_CACHE, ["partnerISO", "cmdCode", "commodity_name", "flowDesc"])
        raw = raw[raw["partnerISO"].isin(PARTNER_ISO_TO_NAME)].copy()
        raw["reporterISO"] = raw["partnerISO"]
        raw["reporterDesc"] = raw["partnerISO"].map(PARTNER_ISO_TO_NAME)
        return raw[GROUP_COLS + ["period", "trade_value", "is_observed"]]
    if not os.path.exists(MONTHLY_CACHE):
        raise FileNotFoundError(
            f"Reporter-side cache '{MONTHLY_CACHE}' not found next to what_if_simulator.py. "
            f"Run prepare_monthly_data.py first, or check the path."
        )
    return load_full_grid(MONTHLY_CACHE, GROUP_COLS)


_SOURCE_CACHE = {}


def _get_source(source):
    if source not in _SOURCE_CACHE:
        _SOURCE_CACHE[source] = _load_source(source)
    return _SOURCE_CACHE[source]


def extract_response_profile(reporter, commodity, flow, event_start, source):
    """Returns the REAL observed % deviation-from-trend curve for one
    historical analog, for months 0..RESPONSE_WINDOW_MONTHS after its
    documented onset. None if there isn't enough data to fit a baseline."""
    try:
        full = _get_source(source)
    except FileNotFoundError:
        raise
    series = full[(full["reporterDesc"] == reporter) & (full["commodity_name"] == commodity)
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
    """Extracts response profiles for every entry in ANALOG_LIBRARY once.

    Returns (profiles_df, skipped_list). Raises RuntimeError if the library
    is empty (strict=True, the default) - a silent empty frame caused the
    dashboard's what-if tab to appear broken with no explanation.

    Call this up front (e.g. at dashboard startup) rather than per-scenario,
    since the historical analogs never change.
    """
    profiles = []
    skipped = []
    for label, r, c, f, start, src in ANALOG_LIBRARY:
        try:
            prof = extract_response_profile(r, c, f, start, src)
        except FileNotFoundError:
            # surface the missing cache immediately - otherwise every mirror
            # analog is silently skipped and the user sees a broken tab
            raise
        if prof is None:
            skipped.append({"analog": label, "reason": "no_usable_baseline_or_series_missing",
                            "reporter": r, "commodity": c, "flow": f, "source": src,
                            "event_start": start})
            continue
        prof = prof.copy()
        prof["analog"] = label
        profiles.append(prof)

    if skipped:
        print(f"NOTE: could not extract a response profile for {len(skipped)} of "
              f"{len(ANALOG_LIBRARY)} historical analogs:")
        for s in skipped:
            print(f"  - {s['analog']} ({s['reporter']} / {s['commodity']} / {s['flow']}, "
                  f"source={s['source']}, event={s['event_start']}): {s['reason']}")

    if not profiles:
        if strict:
            raise RuntimeError(
                f"Analog library is EMPTY: none of the {len(ANALOG_LIBRARY)} historical "
                f"analogs could be extracted. Check that '{MONTHLY_CACHE}' and "
                f"'{MIRROR_CACHE}' exist in the working directory and that their "
                f"columns match GROUP_COLS. Skipped: {[s['analog'] for s in skipped]}"
            )
        return pd.DataFrame(columns=["months_since_onset", "pct_deviation", "analog"]), skipped

    return pd.concat(profiles, ignore_index=True), skipped


def run_scenario(target_reporter, target_commodity, target_flow, hypothetical_start, source,
                  analog_keys=None, analog_profiles=None, skipped_report=None):
    """Project a plausible trend-deviation range for a HYPOTHETICAL disruption.

    target_reporter/commodity/flow: the real series to project.
    hypothetical_start: period int (YYYYMM) where the hypothetical shock begins.
    source: "reporter" or "mirror" - which cache the target series lives in.
    analog_keys: list of ANALOG_LIBRARY labels to use as the basis for the
        projection. Default (None): auto-narrowed to analogs matching the
        TARGET'S OWN commodity. If no same-commodity analog exists, this
        RAISES ValueError rather than silently blending unrelated commodities
        (which produces a near-meaningless band).
    analog_profiles: pass a pre-built build_analog_library_profiles() result
        to avoid recomputing it on every call (e.g. in a dashboard).
    skipped_report: optional list of skipped-analog dicts (from
        build_analog_library_profiles) to surface in warnings.

    Returns (projection_df, analogs_used_df, warnings_list).
    """
    warnings = []
    if skipped_report:
        names = [s["analog"] for s in skipped_report]
        warnings.append(
            f"{len(names)} historical analog(s) could not be extracted and are NOT "
            f"available for this projection: {names}"
        )
    if analog_profiles is None:
        analog_profiles, skipped_report = build_analog_library_profiles()
        if skipped_report:
            warnings.append(
                f"{len(skipped_report)} historical analog(s) could not be extracted: "
                f"{[s['analog'] for s in skipped_report]}"
            )

    if analog_keys is not None:
        analog_profiles = analog_profiles[analog_profiles["analog"].isin(analog_keys)]
        if len(analog_profiles) == 0:
            raise ValueError(
                f"None of the explicitly requested analog_keys "
                f"{analog_keys} are present in the extracted library "
                f"(available: {sorted(analog_profiles['analog'].unique().tolist()) if len(analog_profiles) else []})."
            )
    else:
        # Auto-narrow to analogs of the SAME commodity as the target. Refuse
        # to silently blend unrelated commodities - that default previously
        # produced a near-meaningless band (see module docstring).
        same_commodity_labels = [a[0] for a in ANALOG_LIBRARY if a[2] == target_commodity]
        matched = analog_profiles[analog_profiles["analog"].isin(same_commodity_labels)]
        if len(matched) > 0:
            analog_profiles = matched
            warnings.append(
                f"Auto-selected {matched['analog'].nunique()} analog(s) matching the target commodity "
                f"'{target_commodity}': {matched['analog'].unique().tolist()}. Pass analog_keys explicitly "
                f"to use a different set."
            )
        else:
            raise ValueError(
                f"No historical analog for commodity '{target_commodity}' exists in ANALOG_LIBRARY "
                f"(or all of its analogs failed extraction). Refusing to blend unrelated commodities "
                f"into a projection. Either add a '{target_commodity}' entry to ANALOG_LIBRARY, or "
                f"pass analog_keys explicitly if you really want a cross-commodity illustration."
            )

    if len(analog_profiles) == 0:
        raise ValueError("No analog response data available for the requested analog_keys.")
    n_analogs_used = analog_profiles["analog"].nunique()
    if n_analogs_used < 3:
        warnings.append(
            f"Only {n_analogs_used} historical analog(s) used - the projected range reflects very "
            f"few real cases and should be read as illustrative, not a confidence interval in the "
            f"statistical sense."
        )

    full = _get_source(source)
    target_series = full[(full["reporterDesc"] == target_reporter) & (full["commodity_name"] == target_commodity)
                          & (full["flowDesc"] == target_flow)]
    if len(target_series) == 0:
        raise ValueError(
            f"No data found for {target_reporter} / {target_commodity} / {target_flow} in '{source}' source. "
            f"Check the country/commodity/flow combination and the '{source}' cache."
        )

    target_out = fit_series_baseline(target_series, hypothetical_start, GROUP_COLS)
    if target_out is None:
        raise ValueError(
            f"Not enough observed baseline history before {period_str(hypothetical_start)} "
            f"(need 24+ months) to fit a trend for this series - pick a later hypothetical start "
            f"or a series with longer reporting history."
        )

    target_out = target_out.sort_values("period").reset_index(drop=True)
    baseline_part = target_out[target_out["is_baseline"]]
    model = LinearRegression().fit(baseline_part[["time_idx"]].values, baseline_part["log_value"].values)
    last_idx = baseline_part["time_idx"].max()

    rows = []
    for offset in range(0, RESPONSE_WINDOW_MONTHS + 1):
        future_time_idx = last_idx + 1 + offset
        trend_log = model.predict([[future_time_idx]])[0]
        trend_value = np.expm1(trend_log)

        sub = analog_profiles[analog_profiles["months_since_onset"] == offset]["pct_deviation"]
        if len(sub) == 0:
            continue
        rows.append({
            "months_since_hypothetical_start": offset,
            "projected_trend_value": round(trend_value, 2),
            "low_estimate": round(trend_value * (1 + sub.min() / 100), 2),
            "median_estimate": round(trend_value * (1 + sub.median() / 100), 2),
            "high_estimate": round(trend_value * (1 + sub.max() / 100), 2),
            "n_analogs_at_this_month": len(sub),
        })
    projection = pd.DataFrame(rows)
    analogs_used = analog_profiles[["analog"]].drop_duplicates().reset_index(drop=True)
    return projection, analogs_used, warnings


if __name__ == "__main__":
    # ---------------- DEMO SCENARIO ----------------
    # "What if Ukraine's wheat exports were hit by a new export disruption
    # starting January 2024?" - a real series, a plausible hypothetical
    # (not one already in the analog library), auto-narrowed to the wheat
    # analogs (the only ones a wheat scenario can legitimately use).
    print("Building analog library from documented historical disruptions...")
    profiles, skipped = build_analog_library_profiles()
    print(f"Analog library built: {profiles['analog'].nunique()} usable analogs "
          f"({profiles['analog'].unique().tolist()})")
    if skipped:
        print(f"{len(skipped)} analog(s) skipped - see NOTE lines above.")
    print()

    SCENARIO = dict(
        target_reporter="Ukraine",
        target_commodity="Wheat",
        target_flow="Export",
        hypothetical_start=202401,
        source="reporter",
    )
    print(f"Scenario: what if a disruption hit {SCENARIO['target_reporter']}'s "
          f"{SCENARIO['target_commodity']} {SCENARIO['target_flow'].lower()}s starting "
          f"{period_str(SCENARIO['hypothetical_start'])}?\n")

    try:
        projection, analogs_used, warns = run_scenario(
            **SCENARIO, analog_profiles=profiles, skipped_report=skipped,
        )
        print("Analogs used:", analogs_used["analog"].tolist())
        for w in warns:
            print("WARNING:", w)
        print()
        print(projection.to_string(index=False))
        projection.to_csv("what_if_demo_projection.csv", index=False)
        print("\nSaved: what_if_demo_projection.csv")
    except ValueError as e:
        print("Could not run scenario:", e)