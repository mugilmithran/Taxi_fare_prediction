import os, json
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# --- Spark session ---
spark = (configure_spark_with_delta_pip(
    SparkSession.builder
      .appName("nyc-train-lr-dt-v2")
      .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
      .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
      .config("spark.sql.adaptive.enabled","true")
      .config("spark.sql.shuffle.partitions","16")
)).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 1) Load features
df = spark.read.parquet("data/gold/features_reg_v2").na.drop()
train, test = df.randomSplit([0.8, 0.2], seed=42)

# 2) Assemble features
FEATS = ["trip_distance","log_distance","trip_minutes","passenger_count","hour","is_weekend"]
assembler = VectorAssembler(inputCols=FEATS, outputCol="features_raw")
trainA = assembler.transform(train)
testA  = assembler.transform(test)

# 3) Scale for Linear Regression
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)
scale_model = scaler.fit(trainA)
trainL = scale_model.transform(trainA)
testL  = scale_model.transform(testA)

# --- NEW: save the scaler so batch scoring can reuse it ---
os.makedirs("artifacts/preprocessors", exist_ok=True)
scale_model.write().overwrite().save("artifacts/preprocessors/lin_scaler")

trainL = scale_model.transform(trainA)
testL  = scale_model.transform(testA)

# 4) Models
dt = DecisionTreeRegressor(featuresCol="features_raw", labelCol="label",
                           maxDepth=12, minInstancesPerNode=50, seed=42)
lr = LinearRegression(featuresCol="features", labelCol="label",
                      regParam=0.0, elasticNetParam=0.0)

# 5) Train
dt_model = dt.fit(trainA)
lr_model = lr.fit(trainL)

# 6) Evaluate (RMSE, R2, MAE)
ev_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
ev_r2   = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
ev_mae  = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")

def scores(pred_df):
    return {
        "rmse": ev_rmse.evaluate(pred_df),
        "mae":  ev_mae.evaluate(pred_df),
        "r2":   ev_r2.evaluate(pred_df),
    }

dt_train_pred = dt_model.transform(trainA).select("label","prediction")
dt_test_pred  = dt_model.transform(testA).select("label","prediction")
lr_train_pred = lr_model.transform(trainL).select("label","prediction")
lr_test_pred  = lr_model.transform(testL).select("label","prediction")

dt_train = scores(dt_train_pred); dt_test = scores(dt_test_pred)
lr_train = scores(lr_train_pred); lr_test = scores(lr_test_pred)

print("\n=== DecisionTree ===")
print(f"DT  TRAIN: RMSE={dt_train['rmse']:.3f}, MAE={dt_train['mae']:.3f}, R2={dt_train['r2']:.4f}")
print(f"DT  TEST : RMSE={dt_test['rmse']:.3f}, MAE={dt_test['mae']:.3f}, R2={dt_test['r2']:.4f}")

print("\n=== LinearRegression ===")
print(f"LR  TRAIN: RMSE={lr_train['rmse']:.3f}, MAE={lr_train['mae']:.3f}, R2={lr_train['r2']:.4f}")
print(f"LR  TEST : RMSE={lr_test['rmse']:.3f}, MAE={lr_test['mae']:.3f}, R2={lr_test['r2']:.4f}")

# 7) Save models + metrics
dt_model.write().overwrite().save("artifacts/models/dt_model")
lr_model.write().overwrite().save("artifacts/models/linreg_model")

os.makedirs("artifacts/metrics", exist_ok=True)
with open("artifacts/metrics/metrics_lr_dt_v2.json", "w") as f:
    json.dump({
        "DecisionTreeRegressor": {"train": dt_train, "test": dt_test},
        "LinearRegression":      {"train": lr_train, "test": lr_test},
    }, f, indent=2)

print("\nSaved models to artifacts/models, scaler to artifacts/preprocessors/lin_scaler, and metrics to artifacts/metrics/")
spark.stop()