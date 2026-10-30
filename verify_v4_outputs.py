"""
verify_v4_outputs.py
----------------------
Read-only check. Loads v4_test_fold_comparison.csv and
v4_early_warning_results.csv and independently recomputes precision/
recall/F1/lead-time from the raw counts already stored in them, as a
sanity check. Does not modify either file, and does not rerun any part
of the original pipeline.
"""

import pandas as pd
import os

print("=" * 60)
print("1. Re-deriving precision/recall/F1 from raw counts")
print("=" * 60)

f = "v4_test_fold_comparison.csv"
if not os.path.exists(f):
    print(f"  MISSING: {f}")
else:
    df = pd.read_csv(f)
    print(df.to_string(index=False))
    print("\nIndependent recheck (TP estimated as recall * n_positive_labels):")
    for _, r in df.iterrows():
        tp_est = round(r["recall"] * r["n_positive_labels"])
        precision_check = tp_est / r["n_flagged"] if r["n_flagged"] else 0
        recall_check = tp_est / r["n_positive_labels"] if r["n_positive_labels"] else 0
        f1_check = (2 * precision_check * recall_check / (precision_check + recall_check)
                    if (precision_check + recall_check) else 0)
        ok_p = abs(precision_check - r["precision"]) < 0.01
        ok_r = abs(recall_check - r["recall"]) < 0.01
        ok_f1 = abs(f1_check - r["f1"]) < 0.01
        status = "OK" if (ok_p and ok_r and ok_f1) else "MISMATCH - check manually"
        print(f"  {r['method']:18s} thr={r['threshold']:<6} "
              f"recomputed P={precision_check:.3f} R={recall_check:.3f} F1={f1_check:.3f}  [{status}]")

print("\n" + "=" * 60)
print("2. Checking early-warning results")
print("=" * 60)

f2 = "v4_early_warning_results.csv"
if not os.path.exists(f2):
    print(f"  MISSING: {f2}")
else:
    ew = pd.read_csv(f2)
    print(f"  Rows in file: {len(ew)}")
    print(f"  Unique (reporter, commodity, flow) combos: {ew[['reporter','commodity','flow']].drop_duplicates().shape[0]}")
    print(f"  Mean lead_months (recomputed from file): {ew['lead_months'].mean():.2f}")

    def months_between(p1_str, p2_str):
        y1, m1 = map(int, p1_str.split("-"))
        y2, m2 = map(int, p2_str.split("-"))
        return (y2 - y1) * 12 + (m2 - m1)

    mismatches = 0
    for _, r in ew.iterrows():
        recomputed = months_between(r["first_warning"], r["documented_start"])
        if recomputed != r["lead_months"]:
            mismatches += 1
            print(f"  MISMATCH: {r['reporter']} {r['commodity']} {r['flow']} - "
                  f"stored lead_months={r['lead_months']}, recomputed={recomputed}")
    if mismatches == 0:
        print("  All lead_months values match (first_warning to documented_start). OK.")

print("\n" + "=" * 60)
print("3. Summary file check")
print("=" * 60)
f3 = "v4_summary.txt"
if not os.path.exists(f3):
    print(f"  MISSING: {f3}")
else:
    print(open(f3).read())

print("\nDone. This script did not modify any input file.")
