"""
test_early_warning.py
----------------------
Tests the stretch goal from the abstract: can the model detect a
disruption BEFORE it officially starts, using only data that would
have been available at the time?

Method: for a chosen event, the baseline is cut off several months
BEFORE the official event date (the "lead window"). The model has
NO knowledge of anything after the cutoff. We then check whether the
lead-window months themselves - which are still technically "before"
the war/event - already look anomalous.

If they do, that's real early-warning evidence: the model would have
flagged something unusual using only pre-event information.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

MONTHLY_CACHE = "monthly_series_cache.csv"
MIRROR_CACHE = "mirror_monthly_series_cache.csv"
Z_SCORE_THRESHOLD = 2.5

print("=" * 60)
print("EARLY WARNING TEST")
print("=" * 60)

official_event = input("\nOfficial event date (YYYY-MM), e.g. 2022-02: ").strip()
lead_months = int(input("How many months BEFORE the official date to test as the 'lead window' [6]: ").strip() or 6)

ey, em = map(int, official_event.split("-"))
EVENT_PERIOD = ey * 100 + em


def shift_period(period, months):
    y, m = period // 100, period % 100
    total = y * 12 + (m - 1) + months
    return (total // 12) * 100 + (total % 12 + 1)


LEAD_START = shift_period(EVENT_PERIOD, -lead_months)  # start of the lead window
CUTOFF = LEAD_START  # baseline uses ONLY data strictly before this


def period_str(p):
    return f"{int(p) // 100}-{int(p) % 100:02d}"


print(f"\nOfficial event: {period_str(EVENT_PERIOD)}")
print(f"Lead window being tested: {period_str(LEAD_START)} to {period_str(EVENT_PERIOD - 1) if EVENT_PERIOD % 100 != 1 else period_str(EVENT_PERIOD - 89)}")
print(f"Baseline will ONLY use data before {period_str(CUTOFF)} (as if we didn't know what came after)\n")

country_input = input("Country name, 'all', or 'mirror' for Russia/Belarus via partner data: ").strip()

# ============ LOAD DATA ============

if country_input.lower() == "mirror":
    pdf = pd.read_csv(MIRROR_CACHE)
    pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]
    group_cols = ["partnerISO", "cmdCode", "commodity_name", "flowDesc"]
    label_col = "partnerISO"
else:
    pdf = pd.read_csv(MONTHLY_CACHE)
    pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]
    group_cols = ["reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc"]
    label_col = "reporterDesc"
    if country_input.lower() != "all":
        pdf = pdf[pdf["reporterDesc"].str.lower() == country_input.lower()]

if len(pdf) == 0:
    print("No data found for that filter.")
    exit(0)

all_periods = [y * 100 + m for y in range(2017, 2026) for m in range(1, 13) if y * 100 + m <= 202512]
series_keys = pdf[group_cols].drop_duplicates()
full_index = series_keys.merge(pd.DataFrame({"period": all_periods}), how="cross")
pdf_full = full_index.merge(pdf[group_cols + ["period", "trade_value"]], on=group_cols + ["period"], how="left")
pdf_full["trade_value"] = pdf_full["trade_value"].fillna(0)

# ============ BASELINE USING ONLY PRE-CUTOFF DATA ============

early_warnings = []

for group_key, group_df in pdf_full.groupby(group_cols):
    group_df = group_df.sort_values("period").reset_index(drop=True)
    group_df["time_idx"] = range(len(group_df))

    baseline = group_df[group_df["period"] < CUTOFF].copy()
    lead_window = group_df[(group_df["period"] >= LEAD_START) & (group_df["period"] < EVENT_PERIOD)].copy()

    if len(baseline) < 12 or (baseline["trade_value"] > 0).sum() < 6 or len(lead_window) == 0:
        continue

    baseline["log_value"] = np.log1p(baseline["trade_value"])
    group_df["log_value"] = np.log1p(group_df["trade_value"])

    trend_model = LinearRegression()
    trend_model.fit(baseline[["time_idx"]].values, baseline["log_value"].values)
    group_df["trend_pred"] = trend_model.predict(group_df[["time_idx"]].values)
    group_df["residual"] = group_df["log_value"] - group_df["trend_pred"]

    residual_std = group_df.loc[group_df["period"] < CUTOFF, "residual"].std()
    if residual_std == 0 or np.isnan(residual_std):
        continue

    group_df["z_score"] = group_df["residual"] / residual_std

    lead_rows = group_df[(group_df["period"] >= LEAD_START) & (group_df["period"] < EVENT_PERIOD)]
    flagged = lead_rows[lead_rows["z_score"].abs() > Z_SCORE_THRESHOLD]

    if len(flagged) > 0:
        earliest = flagged.sort_values("period").iloc[0]
        months_early = (EVENT_PERIOD // 100 * 12 + EVENT_PERIOD % 100) - \
                       (int(earliest["period"]) // 100 * 12 + int(earliest["period"]) % 100)
        early_warnings.append({
            "series": " / ".join(str(group_df[c].iloc[0]) for c in [label_col, "commodity_name", "flowDesc"]),
            "first_flagged_month": period_str(earliest["period"]),
            "z_score": round(float(earliest["z_score"]), 2),
            "months_before_official_event": months_early,
        })

# ============ RESULTS ============

print(f"\n{'=' * 60}")
print(f"RESULTS: series showing anomalies in the {lead_months}-month lead window")
print(f"(using ONLY data from before {period_str(CUTOFF)})")
print("=" * 60)

if not early_warnings:
    print(f"\nNo series showed early-warning signals in this window.")
    print("Try a shorter lead window, or check 'mirror' for Russia/Belarus.")
else:
    df_warnings = pd.DataFrame(early_warnings).sort_values("months_before_official_event", ascending=False)
    print(f"\n{len(df_warnings)} series showed anomalies BEFORE the official event date:\n")
    print(df_warnings.to_string(index=False))
    print(f"\nAverage lead time: {df_warnings['months_before_official_event'].mean():.1f} months")
    df_warnings.to_csv("early_warning_results.csv", index=False)
    print("\nSaved early_warning_results.csv")

print("\nNOTE: a flagged lead-window month is tentative early-warning evidence,")
print("not proof the model 'predicted' the event - it means trade patterns")
print("already looked statistically unusual before the event officially began.")
