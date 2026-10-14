import pandas as pd

file_path = "trade_disruption_results_v2.csv"

print("Reading v2 results...")

df = pd.read_csv(file_path)

print("\n===== V2 SERIES CHECK =====")

# Get one row for each country-commodity-flow series
series = df[
    [
        "reporterISO",
        "reporterDesc",
        "cmdCode",
        "commodity_name",
        "flowDesc"
    ]
].drop_duplicates()

print("Total series used by v2:")
print(len(series))

print("\n===== MONTH COVERAGE =====")

months = df.groupby(
    [
        "reporterISO",
        "reporterDesc",
        "cmdCode",
        "commodity_name",
        "flowDesc"
    ]
).size().reset_index(name="months_in_v2")

print("\nSeries with all 108 months:")
print((months["months_in_v2"] == 108).sum())

print("\nSeries with missing months:")
print((months["months_in_v2"] < 108).sum())

print("\nSeries with fewer than 80 months:")
print((months["months_in_v2"] < 80).sum())

print("\n===== MONTH COUNT DISTRIBUTION =====")
print(months["months_in_v2"].value_counts().sort_index())

print("\nDONE")