#!/usr/bin/env python3
import argparse
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import StandardScalerModel

FEATS = ["trip_distance","log_distance","trip_minutes","passenger_count","hour","is_weekend"]

def build_spark():
    return (configure_spark_with_delta_pip(
        SparkSession.builder
          .appName("nyc-batch-score-lr")
          .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
          .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
          .config("spark.sql.adaptive.enabled","true")
          .config("spark.sql.shuffle.partitions","16"))
        .getOrCreate())

def read_any(spark, path, mode_hint):
    path_lower = path.lower()
    if mode_hint == "features":
        # already prepared features parquet/delta dir
        if path_lower.endswith(".csv"):
            # allow CSV of features too
            return (spark.read.option("header","true").option("inferSchema","true").csv(path))
        try:
            return spark.read.format("delta").load(path)
        except Exception:
            return spark.read.parquet(path)
    else:
        # raw CSV/parquet from a user
        if path_lower.endswith(".csv"):
            return (spark.read.option("header","true").option("inferSchema","true").csv(path))
        try:
            return spark.read.format("delta").load(path)
        except Exception:
            return spark.read.parquet(path)

def compute_features_from_raw(df):
    """
    Accepts typical NYC taxi columns and computes the FEATS used by the model.
    Works with either tpep_* or generic pickup/dropoff names.
    """
    # canonical timestamp cols
    pick = F.coalesce(F.col("tpep_pickup_datetime"),
                      F.col("pickup_datetime")).cast("timestamp")
    drop = F.coalesce(F.col("tpep_dropoff_datetime"),
                      F.col("dropoff_datetime")).cast("timestamp")

    # ensure distance & passenger_count exist
    df2 = df
    if "trip_distance" not in df2.columns:
        raise ValueError("Input must contain 'trip_distance'.")
    if "passenger_count" not in df2.columns:
        df2 = df2.withColumn("passenger_count", F.lit(1))

    # engineered
    df2 = (df2
        .withColumn("pickup_ts", pick)
        .withColumn("dropoff_ts", drop)
        .withColumn("trip_minutes",
            F.when(F.col("pickup_ts").isNotNull() & F.col("dropoff_ts").isNotNull(),
                   (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long"))/60.0)
             .otherwise(F.col("trip_minutes")))  # if user already supplied it
        .withColumn("trip_minutes", F.when(F.col("trip_minutes").isNull(), F.lit(12.0)).otherwise(F.col("trip_minutes")))
        .withColumn("hour", F.hour(F.coalesce(F.col("pickup_ts"), F.current_timestamp())))
        .withColumn("dow",  F.dayofweek(F.coalesce(F.col("pickup_ts"), F.current_timestamp())))
        .withColumn("is_weekend", F.when((F.col("dow") == 1) | (F.col("dow") == 7), 1).otherwise(0))
        .withColumn("log_distance", F.log1p(F.col("trip_distance")))
    )
    return df2.select(*FEATS, *[c for c in df2.columns if c not in FEATS])  # keep extra cols for reference

def assemble_and_scale(df, scaler_path):
    assembler = VectorAssembler(inputCols=FEATS, outputCol="features_raw")
    scaled = assembler.transform(df)
    scaler = StandardScalerModel.load(scaler_path)
    return scaler.transform(scaled).withColumnRenamed("features", "features_scaled")

def score(lr_path, df):
    model = LinearRegressionModel.load(lr_path)
    return model.transform(df.withColumnRenamed("features_scaled", "features"))

def write_any(df, out_path, fmt):
    fmt = fmt.lower()
    if fmt == "delta":
        (df.write.format("delta").mode("overwrite").save(out_path))
    elif fmt == "parquet":
        (df.write.mode("overwrite").parquet(out_path))
    elif fmt == "csv":
        (df.coalesce(1)
           .write.mode("overwrite").option("header","true").csv(out_path))
    else:
        raise ValueError("fmt must be delta|parquet|csv")

def main(args):
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # 1) load data
    raw = read_any(spark, args.input, args.mode)

    # 2) ensure we have the six features
    if args.mode == "features":
        need = set(FEATS)
        if not need.issubset(set(map(str.lower, raw.columns))):
            # try case-sensitive check too
            if not need.issubset(set(raw.columns)):
                raise ValueError(f"Input in 'features' mode must contain columns: {FEATS}")
        feats_df = raw
    else:
        feats_df = compute_features_from_raw(raw)

    # Cast to correct types (important for VectorAssembler)
    feats_df = (feats_df
        .withColumn("trip_distance",  F.col("trip_distance").cast("double"))
        .withColumn("log_distance",   F.col("log_distance").cast("double"))
        .withColumn("trip_minutes",   F.col("trip_minutes").cast("double"))
        .withColumn("passenger_count",F.col("passenger_count").cast("int"))
        .withColumn("hour",           F.col("hour").cast("int"))
        .withColumn("is_weekend",     F.col("is_weekend").cast("int"))
    )

    # 3) assemble + scale using training scaler
    assembled = assemble_and_scale(feats_df, args.scaler)

    # 4) predict
    scored = score(args.model, assembled).withColumnRenamed("prediction","fare_pred")

    # 5) write
    cols = [*feats_df.columns, "fare_pred"]
    write_any(scored.select(*cols), args.output, args.fmt)

    print(f"Wrote predictions -> {args.output} ({args.fmt})")
    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch score with Linear Regression")
    p.add_argument("--input",  required=True, help="Path to features table or raw CSV/Parquet/Delta")
    p.add_argument("--mode",   choices=["features","raw"], default="features",
                   help="'features' if input already has engineered columns; 'raw' for typical NYC taxi columns")
    p.add_argument("--output", required=True, help="Output path (folder for delta/parquet/csv)")
    p.add_argument("--fmt",    choices=["delta","parquet","csv"], default="delta")
    p.add_argument("--model",  default="artifacts/models/linreg_model", help="Path to saved LinearRegressionModel")
    p.add_argument("--scaler", default="artifacts/preprocessors/lin_scaler", help="Path to saved StandardScalerModel")
    args = p.parse_args()
    main(args)