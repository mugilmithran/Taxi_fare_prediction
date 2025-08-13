from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.appName("sanity")
         .config("spark.sql.adaptive.enabled","true").getOrCreate())

def qshow(df, n=1): 
    df.show(n, truncate=False)

# --- 1) Check the features table you trained on ---
feat_path = "data/gold/features_reg"  # change to features_reg_v2 if you built it
df = spark.read.parquet(feat_path)

print("\n== Schema ==")
df.printSchema()

print("\n== Row count & nulls per column ==")
total = df.count()
nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
print("rows:", total); qshow(nulls)

print("\n== Basic stats for label & key features ==")
stats = df.select(
    F.mean("label").alias("label_mean"),
    F.stddev("label").alias("label_std"),
    F.min("label").alias("label_min"),
    F.max("label").alias("label_max"),
    F.mean("trip_distance").alias("dist_mean"),
    F.max("trip_distance").alias("dist_max")
)
qshow(stats)

print("\n== Quantiles (median/90/95/99) ==")
q = df.selectExpr(
    "percentile_approx(label, array(0.5,0.9,0.95,0.99)) as label_q",
    "percentile_approx(trip_distance, array(0.5,0.9,0.95,0.99)) as dist_q"
)
qshow(q)

print("\n== Correlations with label ==")
num_cols = [c for c in ["trip_distance","passenger_count","hour","is_weekend","trip_minutes","log_distance"] if c in df.columns]
for c in num_cols:
    corr = df.stat.corr("label", c)
    print(f"corr(label, {c}) = {corr}")

print("\n== A few extreme rows (possible outliers) ==")
print("Top fares:"); qshow(df.orderBy(F.col("label").desc()).select("label","trip_distance","passenger_count","hour","is_weekend").limit(5), 5)
print("Top distances:"); qshow(df.orderBy(F.col("trip_distance").desc()).select("label","trip_distance","passenger_count","hour","is_weekend").limit(5), 5)

# --- 2) (Optional) Peek at Silver to sanity-check duration vs fare ---
try:
    silver = spark.read.format("delta").load("data/silver/trips_clean")
    silver = silver.withColumn("trip_minutes", (F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long"))/60.0)
    print("\n== Silver correlations (fare vs distance/minutes) ==")
    for c in ["trip_distance","trip_minutes"]:
        if c in silver.columns:
            print(f"corr(fare_amount, {c}) =", silver.stat.corr("fare_amount", c))
    qshow(silver.selectExpr(
        "percentile_approx(fare_amount, array(0.5,0.9,0.95,0.99)) as fare_q",
        "percentile_approx(trip_minutes, array(0.5,0.9,0.95,0.99)) as mins_q"
    ))
except Exception as e:
    print("(Silver check skipped):", e)

print("\nSpark UI:", spark.sparkContext.uiWebUrl)
spark.stop()
