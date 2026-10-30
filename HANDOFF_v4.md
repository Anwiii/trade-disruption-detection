# Handoff note — v4 evaluation pass

Save this file. If you hit a message limit, paste it into a new conversation along with `analyze_v4.py` and it has everything needed to pick back up.

## 1. What changed and why

**One new file added: `analyze_v4.py`.** Nothing else was touched — `analyze_trade_disruption.py`, `analyze_trade_disruption_v2.py`, `analyze_mirror_disruption.py`, `explain_anomalies.py`, `dashboard.py`, `prepare_monthly_data.py`, the Hadoop/Spark extraction and merge scripts, and all prior outputs are unchanged. Your original pipeline and results still exist as a baseline to compare against.

**Why `analyze_v4.py` replaces the first `analyze_v3.py` draft:** that draft used `anomaly_event_links.csv` (GDELT matches against the detector's own flagged anomalies) as ground truth. That's circular — positives were pre-selected by the model being evaluated, and series the model didn't flag were never checked, defaulting to "negative" by omission rather than evidence. Recall against it would be trivially inflated. That file should stay in your backup folder, unused.

**What `analyze_v4.py` fixes, in order of your request:**

| # | Issue | Fix |
|---|---|---|
| 1 | Missing months filled with 0 | `is_observed` flag; unobserved months excluded from both baseline fitting and evaluation entirely |
| 2 | Reporting gaps mistaken for trade collapse | Country-month diagnostic flags where <5% of a country's series have any data — separates "this commodity had no trade" from "this country stopped reporting" |
| 3 | Circular ground truth | Replaced with independently dated sanction/event windows (EU/G7 sanction effective dates, not a blanket post-event date) |
| — | Vague labeling | Each label has a real citation-worthy date: Belarus fertilizer sanctions (Jun 2021), EU coal ban (Aug 2022), EU oil ban/price cap (Dec 2022), broader sanctions (Feb 2022) |
| 4 | Only one z-score variant | Added MAD-based robust z-score alongside the original std-based one |
| 5 | Isolation Forest not fairly compared | Same eligible rows, same baseline/eval split, contamination tuned on validation only |
| — | No sustained-change detector | Added CUSUM on the robust z-score |
| — | No early-warning test done properly | Walk-forward: at each candidate date, baseline uses *only* data strictly before that date — no lookahead |
| — | No simple baseline | Added a naive "flag if |% change| exceeds threshold" comparator |
| 5 | No train/val/test discipline | Cross-event split: thresholds tuned on the COVID/wheat event (Mar 2020), frozen, then applied unchanged to the War event (Feb 2022) as held-out test |

## 2. What was actually verified vs. what's pending

**Verified:** the script runs end-to-end without errors, on synthetic data built to match your real CSV's exact column names and types (`reporterISO`, `reporterDesc`, `cmdCode`, `commodity_name`, `flowDesc`, `refYear`, `refMonth`, `trade_value`), including realistic touches — 3% random missing months, a step-change disruption injected into Russia/Belarus fertilizer and energy series at the documented dates, and a COVID-era wheat spike for Russia/Kazakhstan. One bug was caught and fixed this way: the original control-period dates (2018–2019) didn't have 24 months of baseline history before them, given data starts Jan 2017, so every series was skipped (0/0 false-alarm rate, undefined). Fixed to Jul 2019–Jan 2020.

**Not verified: any number from your real dataset.** I do not have your actual `monthly_series_cache.csv`. Every number in the final comparison table below came from synthetic data and must not be quoted in your report. Run it locally and the real table will print in the same format.

## 3. Commands to run

```
# From your BDA folder, with monthly_series_cache.csv already present
# (run prepare_monthly_data.py first if you haven't, or if you've added
# new data since it last ran)

python analyze_v4.py
```

No Spark, no HDFS, no Java switching needed — it reads the local cache only. Takes a few minutes depending on series count.

**Dependencies** (same as your other analysis scripts): `pandas`, `numpy`, `scikit-learn`.

**Expected input:** `monthly_series_cache.csv` in the current folder.

**Expected outputs**, all written to the current folder:
- `v4_reporting_gap_diagnostic.csv` — country-months with near-zero reporting activity
- `v4_validation_fold_sweep.csv` — full threshold sweep on the COVID/wheat fold
- `v4_test_fold_comparison.csv` — **the final comparison table**, all five methods, thresholds frozen from validation, test-fold numbers as-is
- `v4_early_warning_results.csv` — which labeled series were warned early and by how many months
- `v4_summary.txt` — plain-text summary of the key numbers

## 4. Comparison table — SYNTHETIC DATA ONLY, for format reference

Do not cite these numbers. Run locally for real ones.

| method | threshold | precision | recall | f1 | n_flagged | n_positive_labels | n_eligible |
|---|---|---|---|---|---|---|---|
| std | 2.5 | synthetic | synthetic | synthetic | — | — | — |
| robust_MAD | 2.5–3.0 | synthetic | synthetic | synthetic | — | — | — |
| naive_pct_change | tuned | synthetic | synthetic | synthetic | — | — | — |
| isolation_forest | 0.05 (tuned) | synthetic | synthetic | synthetic | — | — | — |
| cusum | tuned | synthetic | synthetic | synthetic | — | — | — |

The real printed table from your run will have this exact structure — paste it back and I'll help interpret it.

## 5. Methodology text (report-ready, describes the method — not synthetic numbers)

> To address weaknesses identified in the initial evaluation — missing months being treated as zero trade, and a ground-truth proxy derived circularly from the detector's own output — the evaluation was redesigned. Months with no reported trade are now distinguished from genuine zero-value trade and excluded from both baseline fitting and evaluation. A country-level reporting-activity diagnostic separately flags country-months where nearly no series report any data, indicating a reporting-system gap rather than a true trade collapse.
>
> Ground-truth labels were rebuilt from independently documented sanction and export-restriction dates (not derived from this model's own flagged anomalies, and not from GDELT), with two separate event folds: the March 2020 COVID-19 wheat export restrictions (Russia, Kazakhstan) as a validation fold, and the February 2022 Russia-Ukraine war sanctions (Belarus, Russia; fertilizer, coal, oil, gas, wheat, each dated to its actual EU/G7 sanction effective date) as a held-out test fold. Detection thresholds for every method — standard z-score, MAD-based robust z-score, Isolation Forest contamination, CUSUM threshold, and a naive percentage-change comparator — were tuned exclusively on the validation fold and then applied unchanged to the test fold, so the reported test-fold metrics are genuinely out-of-sample.
>
> A sustained-change detector (CUSUM) was added alongside the existing point-anomaly detectors to capture persistent deviations that a single-month z-score threshold might under- or over-flag. An early-warning capability was evaluated using a strict walk-forward procedure: for each documented disruption, the baseline was refit using only data available strictly before each candidate date, stepping backward from the disruption's documented start, so that no information from after a candidate date could influence whether that date was flagged. Lead time is reported as the number of months between the first such flag and the documented onset. A false-alarm rate was separately estimated using the same walk-forward procedure applied to a quiet control period (July 2019–January 2020) with no documented disruption.

## 6. Limitations (report-ready, honest)

> Ground-truth labels remain a proxy: documented sanction/restriction dates mark when a disruption became *possible*, not a per-month confirmation that trade was anomalous that specific month. The number of independently labeled positive instances is small (under 15 in the test fold), so precision, recall, and F1 should be read as indicative rather than statistically stable — a small change in which months are labeled can shift these numbers meaningfully. The CUSUM drift parameter was held fixed while only its threshold was tuned; a fuller sweep of both parameters was out of scope for this pass. Early-warning lead time could only be measured for documented events with a specific onset date; it does not establish that the model would generalize to disruptions of a different kind or speed than those observed.

## 7. If you need to resume this specific thread

Tell the next conversation: *"I have analyze_v4.py from a prior session — it fixes circular ground truth, adds MAD z-score, CUSUM, a naive comparator, and walk-forward early warning, validated on synthetic data but not yet run on my real monthly_series_cache.csv. I need help running it and interpreting v4_test_fold_comparison.csv and v4_early_warning_results.csv."* Attach this file and `analyze_v4.py`.
