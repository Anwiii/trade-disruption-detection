from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from pyspark.sql.types import BooleanType

spark = SparkSession.builder \
    .appName("TradeDataAudit") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

file_path = "hdfs://localhost:9000/trade_data/trade_data_final_merged.csv"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(file_path)

print("\n========== SCHEMA ==========")
df.printSchema()

print("\n========== BASIC INFORMATION ==========")
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\n========== YEARS ==========")
df.groupBy("refYear").count().orderBy("refYear").show(50)

print("\n========== REPORTER COUNTRIES ==========")
df.groupBy("reporterISO", "reporterDesc") \
    .count() \
    .orderBy("reporterISO") \
    .show(50, truncate=False)

print("\n========== COMMODITIES ==========")
df.groupBy("cmdCode", "commodity_name") \
    .count() \
    .orderBy("cmdCode") \
    .show(50, truncate=False)

print("\n========== TRADE FLOW ==========")
df.groupBy("flowDesc").count().show()
print("\n========== MISSING VALUES ==========")

from pyspark.sql.types import StringType
from pyspark.sql.functions import col, count, when

string_cols = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, StringType)
]

other_cols = [
    c for c in df.columns
    if c not in string_cols
]

null_counts = df.select(
    [
        count(
            when(col(c).isNull() | (col(c) == ""), c)
        ).alias(c)
        for c in string_cols
    ]
    +
    [
        count(
            when(col(c).isNull(), c)
        ).alias(c)
        for c in other_cols
    ]
)

null_counts.show(truncate=False)
print("\n========== TRADE VALUE SUMMARY ==========")

df.select(
    "primaryValue",
    "fobvalue",
    "cifvalue",
    "qty",
    "netWgt"
).summary().show()

print("\n========== AUDIT COMPLETE ==========")

spark.stop()