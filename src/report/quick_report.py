import os
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import matplotlib.pyplot as plt

# --- Spark (Delta) ---
spark = (configure_spark_with_delta_pip(
    SparkSession.builder
      .appName("quick-report")
      .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
      .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
      .config("spark.sql.adaptive.enabled","true")
)).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --- Load predictions table written in Module 4 ---
pred = spark.read.format("delta").load("data/predictions/lr_features_v2")

# --- Overall metrics ---
ev_rmse = RegressionEvaluator(labelCol="label", predictionCol="fare_pred", metricName="rmse")
ev_mae  = RegressionEvaluator(labelCol="label", predictionCol="fare_pred", metricName="mae")
ev_r2   = RegressionEvaluator(labelCol="label", predictionCol="fare_pred", metricName="r2")

rmse = ev_rmse.evaluate(pred)
mae  = ev_mae.evaluate(pred)
r2   = ev_r2.evaluate(pred)

print(f"\n=== Overall metrics ===\nRMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}\n")

# --- Prep artifacts folder ---
out_dir = "artifacts/report"
os.makedirs(out_dir, exist_ok=True)

# Residual column
pred = pred.withColumn("resid", F.col("label") - F.col("fare_pred")).cache()

# 1) MAE by hour
hourly = (pred.groupBy("hour")
              .agg(F.avg(F.abs(F.col("resid"))).alias("mae"),
                   F.count("*").alias("n"))
              .orderBy("hour"))
pdf_hour = hourly.toPandas()
plt.figure(figsize=(8,4))
plt.bar(pdf_hour["hour"], pdf_hour["mae"])
plt.xlabel("Hour of day"); plt.ylabel("MAE"); plt.title("MAE by Hour")
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "mae_by_hour.png")); plt.close()

# 2) MAE by trip distance bucket (simple bins)
pred_b = pred.withColumn(
    "dist_bin",
    F.when(F.col("trip_distance") < 1, "0–1")\
     .when(F.col("trip_distance") < 3, "1–3")\
     .when(F.col("trip_distance") < 5, "3–5")\
     .when(F.col("trip_distance") < 10, "5–10")\
     .when(F.col("trip_distance") < 20, "10–20")\
     .otherwise("20+")
)
dist = (pred_b.groupBy("dist_bin")
               .agg(F.avg(F.abs(F.col("resid"))).alias("mae"),
                    F.count("*").alias("n"))
               .orderBy(F.when(F.col("dist_bin")=="0–1",0)
                          .when(F.col("dist_bin")=="1–3",1)
                          .when(F.col("dist_bin")=="3–5",2)
                          .when(F.col("dist_bin")=="5–10",3)
                          .when(F.col("dist_bin")=="10–20",4)
                          .otherwise(5)))
pdf_dist = dist.toPandas()
plt.figure(figsize=(7,4))
plt.bar(pdf_dist["dist_bin"], pdf_dist["mae"])
plt.xlabel("Trip distance (mi)"); plt.ylabel("MAE"); plt.title("MAE by Distance Bucket")
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "mae_by_distance.png")); plt.close()

# 3) Residuals histogram (sample for speed)
resid_sample = pred.select("resid").sample(False, 0.02, seed=7).toPandas()
plt.figure(figsize=(7,4))
plt.hist(resid_sample["resid"], bins=60)
plt.xlabel("Residual (label - fare_pred)"); plt.ylabel("Count"); plt.title("Residuals (sample)")
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "residuals_hist.png")); plt.close()

# 4) Fare vs distance scatter (sample)
scatter = pred.select("trip_distance","label","fare_pred").sample(False, 0.01, seed=7).toPandas()
plt.figure(figsize=(7,5))
plt.scatter(scatter["trip_distance"], scatter["label"], s=4, alpha=0.35, label="actual")
plt.scatter(scatter["trip_distance"], scatter["fare_pred"], s=4, alpha=0.35, label="pred")
plt.xlabel("Trip distance (mi)"); plt.ylabel("Fare"); plt.title("Fare vs Distance (sample)")
plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fare_vs_distance_scatter.png")); plt.close()

print("Saved report images to:", out_dir)
spark.stop()