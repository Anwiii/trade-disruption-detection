import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timezone

# 1. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["trade_disruption_db"]
collection = db["monthly_trade_series"]

# 2. Load CSV
print("Loading CSV...")
df = pd.read_csv("monthly_series_cache.csv")

# 3. Convert year and month into a proper timestamp
df["timestamp"] = pd.to_datetime(
    dict(
        year=df["refYear"],
        month=df["refMonth"],
        day=1
    )
)

# 4. Prepare documents for MongoDB
print("Preparing documents...")

data_dict = []

for _, row in df.iterrows():
    document = {
        "timestamp": row["timestamp"].to_pydatetime().replace(
            tzinfo=timezone.utc
        ),
        "metadata": {
            "reporterISO": row["reporterISO"],
            "reporterDesc": row["reporterDesc"],
            "cmdCode": int(row["cmdCode"]),
            "commodity_name": row["commodity_name"],
            "flowDesc": row["flowDesc"]
        },
        "trade_value": float(row["trade_value"])
    }

    data_dict.append(document)

# 5. Insert into MongoDB
print("Inserting data into MongoDB...")

if collection.count_documents({}) > 0:
    print("Collection already contains data.")
    print("Import cancelled to prevent duplicate records.")
else:
    collection.insert_many(data_dict)

    print(f"Successfully inserted {len(data_dict)} records!")

# 6. Close connection
client.close()