import pandas as pd
pdf = pd.read_csv("monthly_series_cache.csv")
pdf["period"] = pdf["refYear"] * 100 + pdf["refMonth"]

check_list = [
    ("BLR", "Sunflower_oil", "Import"),
    ("VNM", "Aluminum", "Import"),
    ("CHN", "Crude_oil", "Import"),
    ("VNM", "Natural_gas", "Import"),
]

for iso, commodity, flow in check_list:
    series = pdf[
        (pdf["reporterISO"] == iso)
        & (pdf["commodity_name"] == commodity)
        & (pdf["flowDesc"] == flow)
    ].sort_values("period")
    last_5 = series.tail(5)[["period", "trade_value"]]
    print(f"\n{iso} | {commodity} | {flow}")
    print(last_5.to_string(index=False))
    print(f"  Total rows in cache: {len(series)} (should be up to 108 if fully reported)")