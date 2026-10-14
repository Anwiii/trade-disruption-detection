from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Check Mirror Data") \
    .getOrCreate()

input_path = "hdfs://localhost:9000/trade_data/trade_data_final_merged.csv"

print("Reading data from HDFS...")

df = spark.read.csv(
    input_path,
    header=True,
    inferSchema=True
)

print("\n===== RUSSIA / BELARUS PARTNER CHECK =====")

russia = df.filter(col("partnerISO") == "RUS")
belarus = df.filter(col("partnerISO") == "BLR")

print("Records with Russia as partner:")
print(russia.count())

print("Records with Belarus as partner:")
print(belarus.count())

print("\nRussia records by year:")
russia.groupBy("refYear").count().orderBy("refYear").show()

print("\nBelarus records by year:")
belarus.groupBy("refYear").count().orderBy("refYear").show()

spark.stop()

print("\nDONE")