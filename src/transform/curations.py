# src/transform/curations.py
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F

builder = (
    SparkSession.builder.appName("nyc-curations")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions","64")               # gentler
    .config("spark.sql.files.maxPartitionBytes","64m")         # smaller tasks
    .config("spark.driver.memory","6g")                        # more heap
    .config("spark.executor.memory","6g")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

bronze = spark.read.format("delta").load("data/bronze/nyc_taxi")

df = (
    bronze
    .withColumn("tpep_pickup_datetime",  F.to_timestamp("tpep_pickup_datetime"))
    .withColumn("tpep_dropoff_datetime", F.to_timestamp("tpep_dropoff_datetime"))
    .withColumn("passenger_count", F.col("passenger_count").cast("int"))
    .withColumn("trip_distance",   F.col("trip_distance").cast("double"))
    .withColumn("fare_amount",     F.col("fare_amount").cast("double"))
    .withColumn("total_amount",    F.col("total_amount").cast("double"))
    .withColumn("PULocationID",    F.col("PULocationID").cast("int"))
    .withColumn("DOLocationID",    F.col("DOLocationID").cast("int"))
)

clean = (
    df.filter(F.col("tpep_pickup_datetime").isNotNull() & F.col("tpep_dropoff_datetime").isNotNull())
      .filter(F.col("trip_distance") > 0)
      .filter(F.col("total_amount").isNotNull())
      .withColumn("trip_minutes", (F.col("tpep_dropoff_datetime").cast("long")
                                   - F.col("tpep_pickup_datetime").cast("long"))/60.0)
      .filter(F.col("trip_minutes") > 0)
      .withColumn("hour", F.hour("tpep_pickup_datetime"))
      .withColumn("dow",  F.date_format("tpep_pickup_datetime","E"))
      .withColumn("ym",   F.date_format("tpep_pickup_datetime","yyyy-MM"))
)

pk = ["VendorID","tpep_pickup_datetime","tpep_dropoff_datetime","PULocationID","DOLocationID","passenger_count"]
clean = clean.dropDuplicates(pk)

# Write Silver: avoid big repartition; lay out by month using partitionBy
(clean.write
   .format("delta")
   .mode("overwrite")
   .partitionBy("ym")
   .option("mergeSchema","true")
   .save("data/silver/trips_clean"))

# Aggregations (Gold)
clean.createOrReplaceTempView("trips")

hourly = spark.sql("""
  SELECT hour, COUNT(*) trips, AVG(trip_distance) avg_distance, AVG(total_amount) avg_total
  FROM trips GROUP BY hour ORDER BY hour
""")
top_pu = spark.sql("""
  SELECT PULocationID, COUNT(*) trips
  FROM trips GROUP BY PULocationID ORDER BY trips DESC LIMIT 50
""")
wday_vs_wend = spark.sql("""
  SELECT CASE WHEN dow IN ('Sat','Sun') THEN 'weekend' ELSE 'weekday' END day_type,
         COUNT(*) trips, AVG(total_amount) avg_total
  FROM trips
  GROUP BY CASE WHEN dow IN ('Sat','Sun') THEN 'weekend' ELSE 'weekday' END
""")

hourly.write.mode("overwrite").parquet("data/gold/hourly_stats")
top_pu.write.mode("overwrite").parquet("data/gold/top_pu_zones")
wday_vs_wend.write.mode("overwrite").parquet("data/gold/wday_vs_wend")

print("✅ Silver: data/silver/trips_clean")
print("✅ Gold: hourly_stats, top_pu_zones, wday_vs_wend")
spark.stop()