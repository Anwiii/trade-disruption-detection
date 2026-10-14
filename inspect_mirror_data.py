from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Inspect Mirror Data") \
    .getOrCreate()

input_path = "hdfs://localhost:9000/trade_data/mirror_data"

print("Reading mirror data from HDFS...")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("\n===== BASIC INFORMATION =====")

print("Total records:")
print(df.count())

print("\nReporters:")
print(df.select("reporterISO").distinct().count())

print("\nCommodities:")
print(df.select("commodity_name").distinct().count())

print("\nYears:")
df.select("refYear").distinct().orderBy("refYear").show()

print("\n===== RECORDS BY PARTNER =====")

df.groupBy("partnerISO") \
    .count() \
    .orderBy("partnerISO") \
    .show()

print("\n===== RECORDS BY FLOW =====")

df.groupBy("flowDesc") \
    .count() \
    .show()

print("\n===== COMMODITIES =====")

df.groupBy("commodity_name") \
    .count() \
    .orderBy(col("count").desc()) \
    .show(20, truncate=False)

print("\n===== TOP REPORTERS =====")

df.groupBy("reporterDesc") \
    .count() \
    .orderBy(col("count").desc()) \
    .show(20, truncate=False)

spark.stop()

print("\nDONE")