from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import glob
import math

# --- Spark + Delta session with more memory and gentler shuffles ---
builder = (
    SparkSession.builder.appName("nyc-ingest")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "96")           # was 200
    .config("spark.default.parallelism", "96")
    .config("spark.driver.memory", "6g")                    # bump memory
    .config("spark.executor.memory", "6g")
    .config("spark.sql.files.maxPartitionBytes", "64m")     # smaller tasks
    .config("spark.sql.parquet.compression.codec", "snappy")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Collect all parquet paths (2021 + 2022)
paths = sorted(
    glob.glob("data/nyc/yellow_tripdata_2021-*.parquet") +
    glob.glob("data/nyc/yellow_tripdata_2022-*.parquet")
)
assert paths, "No parquet files found under data/nyc/."

# Process in batches (e.g., 4 files per batch)
batch_size = 4
num_batches = math.ceil(len(paths) / batch_size)
total_rows = 0

for i in range(num_batches):
    batch = paths[i * batch_size:(i + 1) * batch_size]
    print(f"Reading batch {i+1}/{num_batches}: {len(batch)} files")
    df = spark.read.parquet(*batch)

    # Harmonize vendor col if needed
    if "vendor_id" in df.columns and "VendorID" not in df.columns:
        df = df.withColumnRenamed("vendor_id", "VendorID")

    # Reduce partitions before write to lower memory pressure
    df = df.coalesce(24)  # tune 16/24/32 based on RAM

    # Append to Delta with schema merge
    (df.write
       .format("delta")
       .mode("append")
       .option("mergeSchema", "true")   # <-- handle drift
       .save("data/bronze/nyc_taxi"))

    c = df.count()
    total_rows += c
    print(f"✓ Wrote batch {i+1}, rows={c}, cumulative={total_rows}")

print("Bronze Delta written to data/bronze/nyc_taxi")

# RDD demo on the Bronze table
bronze_df = spark.read.format("delta").load("data/bronze/nyc_taxi")
cols = bronze_df.columns

if "VendorID" in cols:
    df_v = (bronze_df.select("VendorID")
            .na.drop()
            .repartition(64))  # keep it moderate
    vendor_counts = (df_v.rdd
                       .map(lambda r: (r[0], 1))
                       .reduceByKey(lambda a, b: a + b)
                       .collect())
    print("Vendor counts (RDD):", sorted(vendor_counts))
else:
    print("VendorID not found; skipping RDD demo.")

spark.stop()