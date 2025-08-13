# src/qa/sanity_checks_pred.py
import os
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import QuantileDiscretizer

# --- Spark session (Delta + AQE; light shuffles) ---
spark = (
    configure_spark_with_delta_pip(
        SparkSession.builder
        .appName("sanity-pred")
        .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.adaptive.enabled","true")
        .config("spark.sql.shuffle.partitions","16")
    )
).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --- Load predictions table ---
pred = spark.read.format("delta").load("data/predictions/lr_features_v2")

# Optional: sample for speed (override with SANITY_SAMPLE_FRAC=1.0 to use all rows)
SAMPLE_FRAC = float(os.getenv("SANITY_SAMPLE_FRAC", "0.05"))
if SAMPLE_FRAC < 1.0:
    pred = pred.sample(False, SAMPLE_FRAC, seed=42)

pred = pred.select(
    "trip_distance", "trip_minutes", "hour", "is_weekend", "label", "fare_pred"
).persist()

print("\n== Preview ==")
pred.show(5, truncate=False)
nrows = pred.count()
print("rows used for sanity:", nrows, f"(sample_frac={SAMPLE_FRAC})")

# --- Metrics (using fare_pred as prediction) ---
ev_rmse = RegressionEvaluator(labelCol="label", predictionCol="fare_pred", metricName="rmse")
ev_mae  = RegressionEvaluator(labelCol="label", predictionCol="fare_pred", metricName="mae")
ev_r2   = RegressionEvaluator(labelCol="label", predictionCol="fare_pred", metricName="r2")

rmse = ev_rmse.evaluate(pred)
mae  = ev_mae.evaluate(pred)
r2   = ev_r2.evaluate(pred)
print(f"\nOverall metrics -> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# --- Residuals & summaries ---
pred = pred.withColumn("resid", F.col("label") - F.col("fare_pred"))

print("\n== Residual summary ==")
pred.select(
    F.mean("resid").alias("resid_mean"),
    F.stddev("resid").alias("resid_std"),
    F.expr("percentile_approx(resid, array(0.5,0.9,0.95,0.99), 10000)").alias("resid_quantiles")
).show(truncate=False)

print("\n== Calibration / correlation ==")
pred.select(
    F.corr("fare_pred","label").alias("corr_pred_label"),
    F.avg(F.abs("resid")).alias("MAE_agg"),
    F.sqrt(F.avg(F.col("resid")**2)).alias("RMSE_agg")
).show(truncate=False)

# --- Error slices ---

# 1) by hour
by_hour = pred.groupBy("hour").agg(
    F.count("*").alias("n"),
    F.avg(F.abs("resid")).alias("mae"),
    F.avg("resid").alias("bias")
).orderBy("hour")

print("\n== Error by hour ==")
by_hour.show(24, truncate=False)

# 2) by is_weekend
by_wend = pred.groupBy("is_weekend").agg(
    F.count("*").alias("n"),
    F.avg(F.abs("resid")).alias("mae"),
    F.avg("resid").alias("bias")
).orderBy("is_weekend")

print("\n== Error by is_weekend ==")
by_wend.show(truncate=False)

# 3) by distance bucket (quantiles)
qd = QuantileDiscretizer(
    numBuckets=5, inputCol="trip_distance", outputCol="dist_bucket", relativeError=0.01
)
pred_b = qd.fit(pred).transform(pred)

by_dist = pred_b.groupBy("dist_bucket").agg(
    F.count("*").alias("n"),
    F.avg("trip_distance").alias("avg_dist"),
    F.avg(F.abs("resid")).alias("mae"),
    F.avg("resid").alias("bias")
).orderBy("dist_bucket")

print("\n== Error by distance bucket (low → high) ==")
by_dist.show(truncate=False)

# --- Save small slice reports (CSV) ---
out_dir = "artifacts/metrics/pred_slices"
os.makedirs(out_dir, exist_ok=True)
(by_hour.coalesce(1).write.mode("overwrite").option("header", "true")
    .csv(os.path.join(out_dir, "by_hour")))
(by_wend.coalesce(1).write.mode("overwrite").option("header", "true")
    .csv(os.path.join(out_dir, "by_weekend")))
(by_dist.coalesce(1).write.mode("overwrite").option("header", "true")
    .csv(os.path.join(out_dir, "by_distance")))

print(f"\nSaved slice reports under {out_dir}/")

spark.stop()