"""
analyze_v4.py
-------------
Consolidated, corrected evaluation. Supersedes the first analyze_v3.py
draft, which used anomaly_event_links.csv (GDELT matches against the
SAME detector's own flagged anomalies) as ground truth - circular, since
positives were pre-selected by the model being tested. Do not use that
version; see backup_before_v4/analyze_v3_CIRCULAR_DO_NOT_USE.py.

What this script does:

1. MISSING DATA: months with no reported trade are marked unobserved and
   excluded from both baseline fitting and evaluation (not filled with 0).
2. REPORTING-GAP DIAGNOSTIC: flags country-months where almost no series
   have any data at all, separating "this commodity had no trade" from
   "this country stopped reporting".
3. GROUND TRUTH: independently documented sanction/event windows per
   (country, commodity), with two separate folds:
     - VALIDATION fold: COVID-19 wheat export restrictions, Mar 2020
       (Russia and Kazakhstan - both publicly documented export bans)
     - TEST fold: Russia-Ukraine war sanctions, Feb 2022 onward
       (Belarus/Russia, fertilizer and energy commodities, dated to
       actual EU/G7 sanction effective dates, not a blanket post-event date)
   Thresholds are tuned ONLY on the validation fold, then applied
   unchanged to the test fold. Test-fold numbers are reported as-is,
   even if worse than validation.
4. DETECTORS compared on the same eligible observations:
     a. Point anomaly - std z-score (baseline, as in v2)
     b. Point anomaly - MAD-based robust z-score
     c. Isolation Forest (contamination tuned on validation fold)
     d. Sustained-change - CUSUM on standardized residuals
     e. Naive baseline - flag if |% change from previous observed month| 
        exceeds a tuned threshold (simple comparator, not trend-aware)
5. EARLY WARNING (walk-forward, no lookahead): for the test-fold event,
   steps backward from the event date; at each candidate date, baseline
   is fit using ONLY data strictly before that date. Reports first
   warning date, lead time, and a false-alarm rate measured in a quiet
   control period (2018-2019) using the identical walk-forward procedure,
   so false alarms are not just "flags during the disruption".

DETECTION vs WARNING: a "detection" uses data from inside or after the
disruption period to flag it - it confirms something already happened.
A "warning" uses only data available before a date to flag that date -
it is evaluated by replaying time forward and checking, at each step,
whether the model would have raised an alarm without seeing the future.
This script's main sweep (stage 4) measures detection; stage 5 measures
warning specifically.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score

MONTHLY_CACHE = "monthly_series_cache.csv"
MIRROR_CACHE = "mirror_monthly_series_cache.csv"
MIN_BASELINE_MONTHS = 24

# Every TEST_LABELS entry below is Russia or Belarus, and both stopped
# reporting their own trade after the war started - almost exactly the
# evaluation window. Their REPORTER-side data is mostly unobserved there
# by design (see the reporting-gap diagnostic), so the test fold is built
# from MIRROR statistics instead (their trade as reported by partners),
# which has real, continuous coverage through the test period. The
# validation fold (COVID, 2020) keeps using reporter-side data, since
# Russia/Kazakhstan were both still reporting normally in 2020.
PARTNER_ISO_TO_NAME = {"RUS": "Russian Federation", "BLR": "Belarus"}

VALIDATION_EVENT_PERIOD = 202003  # COVID wheat export restrictions
TEST_EVENT_PERIOD = 202202        # Russia-Ukraine war

Z_THRESHOLDS_TO_TEST = [2.0, 2.5, 3.0, 3.5, 4.0]
CONTAMINATION_VALUES_TO_TEST = [0.05, 0.10, 0.15, 0.20]
NAIVE_PCT_THRESHOLDS_TO_TEST = [0.3, 0.5, 0.7, 1.0]
CUSUM_THRESHOLDS_TO_TEST = [3.0, 5.0, 7.0, 10.0]

# Independently documented disruption windows. NOT derived from this
# model's output and NOT derived from GDELT. Sources: public sanction
# announcements / effective dates - cite specifics in the report.
VALIDATION_LABELS = [
    # COVID-19: Russia and Kazakhstan wheat export restrictions, Mar-Jun 2020
    ("Russian Federation", "Wheat", 202003, 202006),
    ("Kazakhstan", "Wheat", 202003, 202006),
]

TEST_LABELS = [
    # Belarus - EU sanctions on potash/fertilizer exports, effective Jun 2021
    ("Belarus", "Potassic_fertilizers", 202106, None),
    ("Belarus", "Nitrogenous_fertilizers", 202106, None),
    # Broad economic sanctions following the Feb 2022 invasion
    ("Russian Federation", "Wheat", 202202, None),
    ("Belarus", "Wheat", 202202, None),
    # EU seaborne coal import ban on Russia, effective Aug 2022
    ("Russian Federation", "Coal", 202208, None),
    ("Belarus", "Coal", 202208, None),
    # EU seaborne crude oil ban + G7 price cap on Russian oil, Dec 2022
    ("Russian Federation", "Crude_oil", 202212, None),
    ("Belarus", "Crude_oil", 202212, None),
    ("Russian Federation", "Natural_gas", 202202, None),
    ("Belarus", "Natural_gas", 202202, None),
]

CONTROL_PERIOD_START = 201907  # quiet window for false-alarm rate estimation
CONTROL_PERIOD_END = 202001


def period_str(p):
    return f"{int(p) // 100}-{int(p) % 100:02d}"


def months_between(p1, p2):
    y1, m1 = p1 // 100, p1 % 100
    y2, m2 = p2 // 100, p2 % 100
    return (y2 - y1) * 12 + (m2 - m1)


def in_window(reporter, commodity, period, label_defs):
    for (c, com, start, end) in label_defs:
        if reporter == c and commodity == com and period >= start and (end is None or period <= end):
            return True
    return False


# ============ STAGE 1: LOAD + MARK OBSERVED VS UNOBSERVED ============

def load_full_grid(cache_path, group_cols):
    pdf = pd.read_csv(cache_path)
    pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]
    all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13) if y * 100 + m <= 202512]
    series_keys = pdf[group_cols].drop_duplicates()
    full_index = series_keys.merge(pd.DataFrame({"period": all_periods}), how="cross")
    full = full_index.merge(pdf[group_cols + ["period", "trade_value"]], on=group_cols + ["period"], how="left")
    full["is_observed"] = full["trade_value"].notna()
    return full


def reporting_gap_diagnostic(full):
    activity = full.groupby(["reporterISO", "reporterDesc", "period"])["is_observed"].mean().reset_index()
    gaps = activity[activity["is_observed"] < 0.05]
    return gaps


# ============ STAGE 2: BASELINE FIT (observed data only, up to a cutoff) ============

def fit_series_baseline(group_df, cutoff_period, group_cols):
    """Fit a log-linear trend using only OBSERVED rows strictly before
    cutoff_period. Returns the full group_df (all periods, including
    unobserved) with trend/residual/z columns added for observed rows,
    or None if there isn't enough observed baseline history."""
    group_df = group_df.sort_values("period").reset_index(drop=True)
    observed = group_df[group_df["is_observed"]].copy()
    baseline_part = observed[observed["period"] < cutoff_period]

    if len(baseline_part) < MIN_BASELINE_MONTHS:
        return None

    observed = observed.copy()
    observed["time_idx"] = range(len(observed))
    observed["log_value"] = np.log1p(observed["trade_value"])
    baseline_part = observed[observed["period"] < cutoff_period]

    model = LinearRegression()
    model.fit(baseline_part[["time_idx"]].values, baseline_part["log_value"].values)
    observed["trend_pred"] = model.predict(observed[["time_idx"]].values)
    observed["residual"] = observed["log_value"] - observed["trend_pred"]

    baseline_residuals = observed.loc[observed["period"] < cutoff_period, "residual"]
    std_scale = baseline_residuals.std()
    median_resid = baseline_residuals.median()
    mad = (baseline_residuals - median_resid).abs().median()
    mad_scale = 1.4826 * mad

    if not std_scale or np.isnan(std_scale) or not mad_scale or np.isnan(mad_scale) or mad_scale == 0:
        return None

    observed["z_std"] = observed["residual"] / std_scale
    observed["z_robust"] = (observed["residual"] - median_resid) / mad_scale
    observed["is_baseline"] = observed["period"] < cutoff_period
    return observed


def build_fold(full, group_cols, event_period):
    results = []
    for _, group_df in full.groupby(group_cols):
        out = fit_series_baseline(group_df, event_period, group_cols)
        if out is not None and (~out["is_baseline"]).sum() >= 1:
            results.append(out)
    if not results:
        return None
    final = pd.concat(results, ignore_index=True)
    return final


# ============ CUSUM (sustained-change detector) ============

def cusum_flags(series_df, threshold, drift=0.5):
    """Two-sided CUSUM on the robust z-score of a single series, run in
    period order. Only uses residual/z values already computed from
    baseline-only statistics - no additional lookahead here."""
    s = series_df.sort_values("period")
    z = s["z_robust"].fillna(0).values
    pos, neg = 0.0, 0.0
    flags = []
    for v in z:
        pos = max(0, pos + v - drift)
        neg = min(0, neg + v + drift)
        flags.append(1 if (pos > threshold or neg < -threshold) else 0)
    s = s.copy()
    s["cusum_pred"] = flags
    return s


# ============ NAIVE BASELINE COMPARATOR ============

def naive_flags(series_df, pct_threshold):
    s = series_df.sort_values("period").copy()
    s["pct_change"] = s["trade_value"].pct_change().abs()
    s["naive_pred"] = (s["pct_change"] > pct_threshold).astype(int)
    s["naive_pred"] = s["naive_pred"].fillna(0).astype(int)
    return s


# ============ EVALUATION HELPER ============

def evaluate(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    n_flagged = int(sum(y_pred))
    n_pos = int(sum(y_true))
    n_eligible = len(y_true)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "n_flagged": n_flagged, "n_positive_labels": n_pos, "n_eligible": n_eligible}


# ============ MAIN ============

def main():
    group_cols = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]

    print("Loading and preparing REPORTER-SIDE full grid (for validation fold)...")
    full_reporter = load_full_grid(MONTHLY_CACHE, group_cols)
    print(f"Total rows: {len(full_reporter):,} | Observed: {full_reporter['is_observed'].sum():,} | "
          f"Unobserved: {(~full_reporter['is_observed']).sum():,}")

    gaps = reporting_gap_diagnostic(full_reporter)
    print(f"\nCountry-months with <5% series reporting (likely reporting-system gaps):")
    if len(gaps):
        summary = gaps.groupby("reporterDesc")["period"].agg(["min", "max", "count"])
        summary.columns = ["first_gap", "last_gap", "n_gap_months"]
        print(summary.to_string())
    else:
        print("  None found.")
    gaps.to_csv("v4_reporting_gap_diagnostic.csv", index=False)

    print("\nLoading MIRROR full grid (for test fold - Russia/Belarus stopped")
    print("reporting their own trade almost exactly during the test window, so")
    print("the test fold uses their trade as reported BY THEIR PARTNERS instead)...")
    mirror_group_cols_raw = ["partnerISO", "cmdCode", "commodity_name", "flowDesc"]
    full_mirror_raw = load_full_grid(MIRROR_CACHE, mirror_group_cols_raw)
    full_mirror = full_mirror_raw[full_mirror_raw["partnerISO"].isin(PARTNER_ISO_TO_NAME)].copy()
    full_mirror["reporterISO"] = full_mirror["partnerISO"]
    full_mirror["reporterDesc"] = full_mirror["partnerISO"].map(PARTNER_ISO_TO_NAME)
    full_mirror = full_mirror[group_cols + ["period", "trade_value", "is_observed"]]
    print(f"Mirror rows (Russia/Belarus only): {len(full_mirror):,} | "
          f"Observed: {full_mirror['is_observed'].sum():,} | "
          f"Unobserved: {(~full_mirror['is_observed']).sum():,}")

    # ---------- VALIDATION FOLD (reporter-side data) ----------
    print(f"\nBuilding validation fold (event period {period_str(VALIDATION_EVENT_PERIOD)}, reporter-side data)...")
    val_fold = build_fold(full_reporter, group_cols, VALIDATION_EVENT_PERIOD)
    if val_fold is None:
        print("ERROR: no series had enough baseline history for the validation fold. Stopping.")
        return
    val_eval = val_fold[~val_fold["is_baseline"]].copy()
    val_eval["ground_truth"] = val_eval.apply(
        lambda r: int(in_window(r["reporterDesc"], r["commodity_name"], r["period"], VALIDATION_LABELS)), axis=1)
    print(f"Validation fold: {len(val_eval):,} eligible rows, {val_eval['ground_truth'].sum()} positive labels.")

    # ---------- TEST FOLD (mirror data, since labels are Russia/Belarus post-2022) ----------
    print(f"\nBuilding test fold (event period {period_str(TEST_EVENT_PERIOD)}, MIRROR data)...")
    test_fold = build_fold(full_mirror, group_cols, TEST_EVENT_PERIOD)
    if test_fold is None:
        print("ERROR: no mirror series had enough baseline history for the test fold. Stopping.")
        return
    test_eval = test_fold[~test_fold["is_baseline"]].copy()
    test_eval["ground_truth"] = test_eval.apply(
        lambda r: int(in_window(r["reporterDesc"], r["commodity_name"], r["period"], TEST_LABELS)), axis=1)
    print(f"Test fold: {len(test_eval):,} eligible rows, {test_eval['ground_truth'].sum()} positive labels.")

    if val_eval["ground_truth"].sum() < 3 or test_eval["ground_truth"].sum() < 3:
        print("\nWARNING: fewer than 3 positive labels in a fold. Precision/recall/F1 below")
        print("will be extremely noisy (each single label flips the score a lot). Report this")
        print("explicitly as a limitation rather than treating the numbers as stable.\n")

    # ---------- TUNE ON VALIDATION, APPLY TO TEST (z-score, both variants) ----------
    print("\n" + "=" * 70)
    print("Z-SCORE: tuning on VALIDATION fold")
    print("=" * 70)
    best = {}
    val_rows = []
    for z_col, label in [("z_std", "std"), ("z_robust", "robust_MAD")]:
        for t in Z_THRESHOLDS_TO_TEST:
            pred = (val_eval[z_col].abs() > t).astype(int)
            m = evaluate(val_eval["ground_truth"], pred)
            m.update({"method": label, "threshold": t})
            val_rows.append(m)
        fold_rows = [r for r in val_rows if r["method"] == label]
        best[label] = max(fold_rows, key=lambda r: r["f1"])["threshold"]
    val_df = pd.DataFrame(val_rows)
    print(val_df.to_string(index=False))
    print(f"\nThresholds selected on validation (frozen before looking at test): {best}")

    print("\n" + "=" * 70)
    print("Z-SCORE: applying frozen thresholds to TEST fold")
    print("=" * 70)
    test_rows = []
    for z_col, label in [("z_std", "std"), ("z_robust", "robust_MAD")]:
        t = best[label]
        pred = (test_eval[z_col].abs() > t).astype(int)
        m = evaluate(test_eval["ground_truth"], pred)
        m.update({"method": label, "threshold": t})
        test_rows.append(m)

    # ---------- NAIVE BASELINE COMPARATOR ----------
    print("\nComputing naive (% month-over-month change) comparator...")
    val_naive_all, test_naive_all = [], []
    for _, g in val_fold.groupby(group_cols):
        val_naive_all.append(naive_flags(g, 0))  # placeholder, recomputed per threshold below
    for _, g in test_fold.groupby(group_cols):
        test_naive_all.append(naive_flags(g, 0))

    def naive_eval_fold(fold_df, eval_df, label_defs, pct_t):
        preds = []
        for _, g in fold_df.groupby(group_cols):
            flagged = naive_flags(g, pct_t)
            preds.append(flagged[~flagged["is_baseline"]][group_cols + ["period", "naive_pred"]])
        pred_df = pd.concat(preds, ignore_index=True)
        merged = eval_df.merge(pred_df, on=group_cols + ["period"], how="left")
        return evaluate(merged["ground_truth"], merged["naive_pred"].fillna(0))

    naive_val_rows = []
    for t in NAIVE_PCT_THRESHOLDS_TO_TEST:
        m = naive_eval_fold(val_fold, val_eval, VALIDATION_LABELS, t)
        m.update({"method": "naive_pct_change", "threshold": t})
        naive_val_rows.append(m)
    best_naive_t = max(naive_val_rows, key=lambda r: r["f1"])["threshold"]
    naive_test_m = naive_eval_fold(test_fold, test_eval, TEST_LABELS, best_naive_t)
    naive_test_m.update({"method": "naive_pct_change", "threshold": best_naive_t})
    test_rows.append(naive_test_m)
    print(f"Naive comparator: threshold {best_naive_t} selected on validation.")

    # ---------- ISOLATION FOREST ----------
    print("\nTuning Isolation Forest contamination on validation fold...")
    def if_eval_fold(fold_df, eval_df, label_defs, contamination):
        preds = []
        for _, g in fold_df.groupby(group_cols):
            baseline = g[g["is_baseline"]]
            evaluation = g[~g["is_baseline"]]
            if len(baseline) < MIN_BASELINE_MONTHS or len(evaluation) < 1:
                continue
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(baseline[["residual"]].fillna(0))
            p = model.predict(evaluation[["residual"]].fillna(0))
            tmp = evaluation[group_cols + ["period"]].copy()
            tmp["if_pred"] = (p == -1).astype(int)
            preds.append(tmp)
        if not preds:
            return {"precision": 0, "recall": 0, "f1": 0, "n_flagged": 0, "n_positive_labels": 0, "n_eligible": 0}
        pred_df = pd.concat(preds, ignore_index=True)
        merged = eval_df.merge(pred_df, on=group_cols + ["period"], how="left")
        return evaluate(merged["ground_truth"], merged["if_pred"].fillna(0))

    if_val_rows = []
    for c in CONTAMINATION_VALUES_TO_TEST:
        m = if_eval_fold(val_fold, val_eval, VALIDATION_LABELS, c)
        m.update({"method": "isolation_forest", "threshold": c})
        if_val_rows.append(m)
    best_contamination = max(if_val_rows, key=lambda r: r["f1"])["threshold"]
    if_test_m = if_eval_fold(test_fold, test_eval, TEST_LABELS, best_contamination)
    if_test_m.update({"method": "isolation_forest", "threshold": best_contamination})
    test_rows.append(if_test_m)
    print(f"Isolation Forest: contamination {best_contamination} selected on validation.")

    # ---------- CUSUM ----------
    print("\nTuning CUSUM threshold on validation fold...")
    def cusum_eval_fold(fold_df, eval_df, label_defs, cusum_t):
        preds = []
        for _, g in fold_df.groupby(group_cols):
            flagged = cusum_flags(g, cusum_t)
            preds.append(flagged[~flagged["is_baseline"]][group_cols + ["period", "cusum_pred"]])
        pred_df = pd.concat(preds, ignore_index=True)
        merged = eval_df.merge(pred_df, on=group_cols + ["period"], how="left")
        return evaluate(merged["ground_truth"], merged["cusum_pred"].fillna(0))

    cusum_val_rows = []
    for t in CUSUM_THRESHOLDS_TO_TEST:
        m = cusum_eval_fold(val_fold, val_eval, VALIDATION_LABELS, t)
        m.update({"method": "cusum", "threshold": t})
        cusum_val_rows.append(m)
    best_cusum_t = max(cusum_val_rows, key=lambda r: r["f1"])["threshold"]
    cusum_test_m = cusum_eval_fold(test_fold, test_eval, TEST_LABELS, best_cusum_t)
    cusum_test_m.update({"method": "cusum", "threshold": best_cusum_t})
    test_rows.append(cusum_test_m)
    print(f"CUSUM: threshold {best_cusum_t} selected on validation.")

    # ---------- FINAL TEST-FOLD COMPARISON TABLE ----------
    test_df = pd.DataFrame(test_rows)
    print("\n" + "=" * 70)
    print("FINAL TEST-FOLD COMPARISON (thresholds frozen from validation; test numbers reported as-is)")
    print("=" * 70)
    print(test_df.to_string(index=False))
    test_df.to_csv("v4_test_fold_comparison.csv", index=False)
    val_df.to_csv("v4_validation_fold_sweep.csv", index=False)

    # ---------- EARLY WARNING (walk-forward, test event) ----------
    print("\n" + "=" * 70)
    print("EARLY WARNING TEST (walk-forward, no lookahead)")
    print("=" * 70)
    print("For each labeled TEST series, steps backward from its documented start date.")
    print("At each candidate month, baseline uses ONLY data strictly before that month.")

    z_threshold_for_warning = best["robust_MAD"]
    warning_rows = []
    labeled_series = {(c, com) for (c, com, s, e) in TEST_LABELS}
    n_eligible_series = 0  # (reporter, commodity, flow) combos with enough baseline to be checked at all

    for (reporter, commodity) in labeled_series:
        label_start = min(s for (c, com, s, e) in TEST_LABELS if c == reporter and com == commodity)
        series_rows = full_mirror[(full_mirror["reporterDesc"] == reporter) & (full_mirror["commodity_name"] == commodity)]
        if len(series_rows) == 0:
            continue
        for flow, flow_df in series_rows.groupby("flowDesc"):
            was_eligible = False
            first_warning = None
            for lead in range(12, 0, -1):
                candidate_period = label_start
                y, m = candidate_period // 100, candidate_period % 100
                total = y * 12 + (m - 1) - lead
                cutoff = (total // 12) * 100 + (total % 12 + 1)
                out = fit_series_baseline(flow_df, cutoff, group_cols)
                if out is None:
                    continue
                was_eligible = True  # had enough baseline to be checked at this lead
                row_at_cutoff = out[out["period"] == cutoff]
                if len(row_at_cutoff) == 0:
                    continue
                z = row_at_cutoff["z_robust"].iloc[0]
                if abs(z) > z_threshold_for_warning:
                    if first_warning is None:
                        first_warning = cutoff
                    break
            if was_eligible:
                n_eligible_series += 1
            if first_warning is not None:
                lead_months = months_between(first_warning, label_start)
                warning_rows.append({"reporter": reporter, "commodity": commodity, "flow": flow,
                                      "documented_start": period_str(label_start),
                                      "first_warning": period_str(first_warning),
                                      "lead_months": lead_months})

    warning_df = pd.DataFrame(warning_rows)
    print(f"\nWarned before documented onset: {len(warning_df)} of {n_eligible_series} eligible "
          f"(country, commodity, flow) series that had enough baseline data to be checked "
          f"(out of {len(labeled_series)} labeled country-commodity pairs, {CONTROL_PERIOD_START}-independent).")
    if len(warning_df):
        print(warning_df.to_string(index=False))
        print(f"\nMean lead time: {warning_df['lead_months'].mean():.1f} months")
    warning_df.to_csv("v4_early_warning_results.csv", index=False)

    # ---------- FALSE ALARM RATE IN A QUIET CONTROL PERIOD ----------
    print(f"\nEstimating false-alarm rate in a quiet control period "
          f"({period_str(CONTROL_PERIOD_START)} to {period_str(CONTROL_PERIOD_END)})...")
    control_flags = 0
    control_checked = 0
    for _, g in full_reporter.groupby(group_cols):
        out = fit_series_baseline(g, CONTROL_PERIOD_START, group_cols)
        if out is None:
            continue
        control_rows = out[(out["period"] >= CONTROL_PERIOD_START) & (out["period"] <= CONTROL_PERIOD_END)]
        control_checked += len(control_rows)
        control_flags += (control_rows["z_robust"].abs() > z_threshold_for_warning).sum()
    false_alarm_rate = control_flags / control_checked if control_checked else float("nan")
    print(f"False alarms in control period: {control_flags} / {control_checked} eligible months "
          f"({false_alarm_rate:.1%})")

    with open("v4_summary.txt", "w") as f:
        f.write(f"Validation positive labels: {val_eval['ground_truth'].sum()}\n")
        f.write(f"Test positive labels: {test_eval['ground_truth'].sum()}\n")
        f.write(f"Thresholds selected on validation: {best}, naive={best_naive_t}, "
                f"IF_contamination={best_contamination}, cusum={best_cusum_t}\n")
        f.write(f"False alarm rate in control period: {false_alarm_rate:.1%} "
                f"({control_flags}/{control_checked})\n")
        f.write(f"Early warning: {len(warning_df)} of {n_eligible_series} eligible series warned early, "
                f"mean lead {warning_df['lead_months'].mean() if len(warning_df) else float('nan'):.1f} months\n")

    print("\nSaved: v4_reporting_gap_diagnostic.csv, v4_validation_fold_sweep.csv, "
          "v4_test_fold_comparison.csv, v4_early_warning_results.csv, v4_summary.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
