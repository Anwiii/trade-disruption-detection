from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Extract Mirror Data") \
    .getOrCreate()

input_path = "hdfs://localhost:9000/trade_data/trade_data_final_merged.csv"

print("Reading data from HDFS...")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("Filtering Russia and Belarus as partners...")

mirror = df.filter(
    (col("partnerISO") == "RUS") |
    (col("partnerISO") == "BLR")
)

mirror = mirror.select(
    "reporterISO",
    "reporterDesc",
    "partnerISO",
    "cmdCode",
    "commodity_name",
    "flowDesc",
    "refYear",
    "refMonth",
    "primaryValue"
)

print("Writing mirror data...")

output_path = "hdfs://localhost:9000/trade_data/mirror_data"

mirror.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("Mirror data saved to:")
print(output_path)

print("\nRussia records:", mirror.filter(col("partnerISO") == "RUS").count())
print("Belarus records:", mirror.filter(col("partnerISO") == "BLR").count())

spark.stop()

print("\nDONE")