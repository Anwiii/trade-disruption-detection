from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = SparkSession.builder \
    .appName("Check Missing Months") \
    .getOrCreate()

input_path = "hdfs://localhost:9000/trade_data/trade_data_final_merged.csv"

print("Reading data from HDFS...")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

# Keep only the columns needed for this check
df = df.select(
    "reporterISO",
    "reporterDesc",
    "cmdCode",
    "commodity_name",
    "flowDesc",
    "refYear",
    "refMonth"
)

# Find which months actually exist
monthly = df.dropDuplicates([
    "reporterISO",
    "reporterDesc",
    "cmdCode",
    "commodity_name",
    "flowDesc",
    "refYear",
    "refMonth"
])

print("Counting months for each series...")

series_months = monthly.groupBy(
    "reporterISO",
    "reporterDesc",
    "cmdCode",
    "commodity_name",
    "flowDesc"
).agg(
    count("*").alias("months_present")
)

# 2017-01 to 2025-12 = 108 months
series_months = series_months.withColumn(
    "months_missing",
    108 - col("months_present")
)

print("\n===== MISSING MONTH AUDIT =====")

print("Total series:")
print(series_months.count())

print("\nSeries with all 108 months:")
print(
    series_months
    .filter(col("months_missing") == 0)
    .count()
)

print("\nSeries with at least one missing month:")
print(
    series_months
    .filter(col("months_missing") > 0)
    .count()
)

print("\nSeries with more than 20 missing months:")
print(
    series_months
    .filter(col("months_missing") > 20)
    .count()
)

print("\nSeries with more than 50 missing months:")
print(
    series_months
    .filter(col("months_missing") > 50)
    .count()
)

print("\n===== MOST INCOMPLETE SERIES =====")

series_months.orderBy(
    col("months_missing").desc()
).show(20, truncate=False)

print("\n===== MISSING MONTH DISTRIBUTION =====")

series_months.groupBy(
    "months_missing"
).count().orderBy(
    col("months_missing")
).show(30)

spark.stop()

print("\nDONE")