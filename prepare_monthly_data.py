"""
STEP 1 (run once): Aggregate the full HDFS dataset down to monthly
(country, commodity, flow) time series and save locally as a CSV.

This is the heavy Spark/HDFS step. Run it once - after that,
analyze_event.py can analyze ANY event date instantly without
needing to re-read the full 5.1M-row dataset from HDFS every time.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum

HDFS_PATH = "hdfs://localhost:9000/trade_data/trade_data_final_merged.csv"
MIRROR_PATH = "hdfs://localhost:9000/trade_data/mirror_data"
OUTPUT_FILE = "monthly_series_cache.csv"
MIRROR_OUTPUT_FILE = "mirror_monthly_series_cache.csv"

spark = SparkSession.builder.appName("PrepareMonthlySeries").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("Reading full dataset from HDFS...")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(HDFS_PATH)

print("Aggregating to monthly (country, commodity, flow) series...")
monthly = (
    df.groupBy("reporterISO", "reporterDesc", "cmdCode", "commodity_name", "flowDesc", "refYear", "refMonth")
    .agg(spark_sum("primaryValue").alias("trade_value"))
)

print("Saving to local cache:", OUTPUT_FILE)
monthly.toPandas().to_csv(OUTPUT_FILE, index=False)

print("\nReading mirror dataset from HDFS...")
mirror_df = spark.read.option("header", "true").option("inferSchema", "true").csv(MIRROR_PATH)

print("Aggregating mirror data...")
mirror_monthly = (
    mirror_df.groupBy("partnerISO", "cmdCode", "commodity_name", "flowDesc", "refYear", "refMonth")
    .agg(spark_sum("primaryValue").alias("trade_value"))
)

print("Saving to local cache:", MIRROR_OUTPUT_FILE)
mirror_monthly.toPandas().to_csv(MIRROR_OUTPUT_FILE, index=False)

spark.stop()

print("\nDone! You can now run analyze_event.py as many times as you want,")
print("with any event date, without needing Spark/HDFS again.")
