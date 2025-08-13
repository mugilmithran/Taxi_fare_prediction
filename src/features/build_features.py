from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F

spark = (configure_spark_with_delta_pip(
    SparkSession.builder
      .appName("nyc-features-v2")
      .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
      .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
)).getOrCreate()

silver = spark.read.format("delta").load("data/silver/trips_clean")

# Filter obvious errors/outliers (based on your quantiles)
clean = (silver
  .filter((F.col("fare_amount").isNotNull()) & (F.col("fare_amount") > 1) & (F.col("fare_amount") < 200))
  .filter((F.col("trip_distance") > 0.1) & (F.col("trip_distance") < 50))
  .filter((F.col("trip_minutes") > 1) & (F.col("trip_minutes") < 180))
)

features = (clean
  .withColumn("is_weekend", F.when(F.col("dow").isin("Sat","Sun"), 1).otherwise(0))
  .withColumn("log_distance", F.log1p("trip_distance"))
  .select(
      "trip_distance", "log_distance", "trip_minutes",
      "passenger_count", "hour", "is_weekend",
      F.col("fare_amount").alias("label")
  )
  .na.drop()
)

features.write.mode("overwrite").parquet("data/gold/features_reg_v2")
print("Saved data/gold/features_reg_v2")
spark.stop()