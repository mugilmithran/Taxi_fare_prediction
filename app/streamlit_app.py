import os, io, numpy as np, pandas as pd, streamlit as st
import plotly.express as px
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.regression import LinearRegressionModel

st.set_page_config(page_title="NYC Taxi — Fare Predictor", layout="wide")

# ---- Config / defaults ----
MODEL_PATH   = "artifacts/models/linreg_model"
SCALER_PATH  = "artifacts/preprocessors/lin_scaler"
FEATS = ["trip_distance","log_distance","trip_minutes","passenger_count","hour","is_weekend"]

# ---- Spark (cached) ----
@st.cache_resource
def get_spark():
    return (configure_spark_with_delta_pip(
        SparkSession.builder
          .appName("nyc-taxi-ui")
          .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
          .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
          .config("spark.sql.adaptive.enabled","true")
          .config("spark.sql.shuffle.partitions","16")
    ).getOrCreate())

spark = get_spark()
spark.sparkContext.setLogLevel("WARN")

# ---- Load artifacts (cached) ----
@st.cache_resource
def load_artifacts(model_path, scaler_path):
    assembler = VectorAssembler(inputCols=FEATS, outputCol="features_raw")
    scaler = StandardScalerModel.load(scaler_path)
    model  = LinearRegressionModel.load(model_path)
    return assembler, scaler, model

assembler, scaler, model = load_artifacts(MODEL_PATH, SCALER_PATH)

# ---- Feature engineering: accept EITHER schema ----
def compute_features_from_raw(sdf):
    # normalize column names to lowercase for matching
    for c in sdf.columns:
        sdf = sdf.withColumnRenamed(c, c.lower())
    cols = set(sdf.columns)

    # base requirement
    if "trip_distance" not in cols:
        raise ValueError("CSV must include 'trip_distance'.")

    # defaults / optional columns
    if "passenger_count" not in cols:
        sdf = sdf.withColumn("passenger_count", F.lit(1))

    # choose timestamp columns if present
    def first_present(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    ts_pick = first_present(["tpep_pickup_datetime","pickup_datetime","pickup_ts"])
    ts_drop = first_present(["tpep_dropoff_datetime","dropoff_datetime","dropoff_ts"])

    have_hour      = "hour" in cols
    have_is_weekend= "is_weekend" in cols

    # derive hour/is_weekend if not provided but we have a timestamp
    if not (have_hour and have_is_weekend):
        if ts_pick:
            sdf = sdf.withColumn("pickup_ts", F.to_timestamp(F.col(ts_pick)))
            sdf = sdf.withColumn("hour", F.hour("pickup_ts"))
            sdf = sdf.withColumn("dow",  F.dayofweek("pickup_ts"))
            sdf = sdf.withColumn("is_weekend", F.when(F.col("dow").isin(1,7), 1).otherwise(0)).drop("dow")
            have_hour, have_is_weekend = True, True
        else:
            raise ValueError(
                "Provide either columns ['hour','is_weekend'] OR a pickup timestamp "
                "('tpep_pickup_datetime' or 'pickup_datetime')."
            )
    else:
        # cast provided hour/is_weekend
        sdf = sdf.withColumn("hour", F.col("hour").cast("int"))
        sdf = sdf.withColumn("is_weekend", F.col("is_weekend").cast("int"))

    # trip_minutes: use provided, else compute from timestamps, else fallback
    if "trip_minutes" in cols:
        sdf = sdf.withColumn("trip_minutes", F.col("trip_minutes").cast("double"))
    elif ts_pick and ts_drop:
        sdf = sdf.withColumn("pickup_ts",  F.to_timestamp(F.col(ts_pick))) \
                 .withColumn("dropoff_ts", F.to_timestamp(F.col(ts_drop))) \
                 .withColumn("trip_minutes",
                     (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long"))/60.0
                 )
    else:
        # sensible default if duration not available
        sdf = sdf.withColumn("trip_minutes", F.lit(12.0))

    # final casts & engineered columns
    sdf = (sdf
        .withColumn("trip_distance",   F.col("trip_distance").cast("double"))
        .withColumn("log_distance",    F.log1p(F.col("trip_distance")))
        .withColumn("passenger_count", F.col("passenger_count").cast("int"))
        .withColumn("hour",            F.col("hour").cast("int"))
        .withColumn("is_weekend",      F.col("is_weekend").cast("int"))
    )
    # optional passthrough of actual fare if provided by the CSV
    if "trip_fare" in sdf.columns:
        sdf = sdf.withColumn("trip_fare", F.col("trip_fare").cast("double"))

    base_cols = FEATS
    extra = ["trip_fare"] if "trip_fare" in sdf.columns else []
    return sdf.select(*base_cols, *extra)

def score(sdf_feats):
    assembled = assembler.transform(sdf_feats)
    scaled    = scaler.transform(assembled).withColumnRenamed("features","features_scaled")
    scored    = model.transform(scaled.withColumnRenamed("features_scaled","features")) \
                   .withColumnRenamed("prediction","fare_pred")
    return scored

# ---- UI ----
st.title("NYC Taxi — Fare Predictor")

# Upload & score
st.subheader("Upload CSV to get fare predictions")
st.caption(
    "Accepted schemas — "
    "**A)** `trip_distance` (+ optional `passenger_count`) and a pickup timestamp "
    "`tpep_pickup_datetime` or `pickup_datetime` (optional dropoff timestamp for better `trip_minutes`); "
    "**B)** `trip_distance`, `trip_minutes`, `passenger_count`, `hour`, `is_weekend`."
)
file = st.file_uploader("Choose a CSV", type=["csv"])

if file is not None:
    try:
        pdf = pd.read_csv(file)
        st.write(f"Uploaded rows: {len(pdf):,}")
        sdf = spark.createDataFrame(pdf)
        feats = compute_features_from_raw(sdf)
        scored = score(feats)

        select_cols = FEATS + ["fare_pred"]
        if "trip_fare" in scored.columns:
            scored = scored.withColumn("abs_error", F.abs(F.col("trip_fare") - F.col("fare_pred")))
            select_cols = FEATS + ["trip_fare", "fare_pred", "abs_error"]

        outpdf = scored.select(*select_cols).limit(10000).toPandas()

        st.success("Scored successfully. Preview below:")
        st.dataframe(outpdf.head(200), use_container_width=True)

        buf = io.StringIO()
        outpdf.to_csv(buf, index=False)
        st.download_button("Download predictions CSV", data=buf.getvalue(),
                           file_name="predictions.csv", mime="text/csv")

        # ----- Visualizations (logical & easy to read) -----
        st.subheader("Visualizations")
        st.caption("Quick read: These charts summarize your upload. If you included an actual `trip_fare`, we show error-based views (MAE, residuals). Otherwise we summarize the pattern of predicted fares by hour and by distance.")

        has_actual = "trip_fare" in outpdf.columns

        # Metrics on uploaded data (if actual provided)
        if has_actual:
            err = outpdf["trip_fare"] - outpdf["fare_pred"]
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            denom = float(np.sum((outpdf["trip_fare"] - outpdf["trip_fare"].mean())**2))
            r2 = float(1 - (np.sum(err**2) / denom)) if denom > 0 else float("nan")

            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{rmse:.2f}")
            m2.metric("MAE",  f"{mae:.2f}")
            m3.metric("R²",   f"{r2:.3f}")

        # By hour of day
        if "hour" in outpdf.columns:
            st.markdown("**By hour of day**")
            if has_actual:
                hour_mae = (outpdf
                            .groupby("hour")
                            .apply(lambda g: (g["trip_fare"] - g["fare_pred"]).abs().mean())
                            .reset_index(name="MAE"))
                fig_hour_mae = px.bar(hour_mae, x="hour", y="MAE",
                                      title="MAE by Hour of Day",
                                      labels={"hour":"Hour of day","MAE":"Mean Absolute Error"},
                                      color_discrete_sequence=["#1f77b4"])
                st.plotly_chart(fig_hour_mae, use_container_width=True)
                st.caption("This bar chart shows average absolute prediction error for each pickup hour. Taller bars indicate hours where the model struggles more.")
            else:
                hour_pred = (outpdf.groupby("hour")["fare_pred"].mean()
                                     .reset_index(name="Average predicted fare ($)"))
                fig_hour_pred = px.bar(hour_pred, x="hour", y="Average predicted fare ($)",
                                       title="Average Predicted Fare by Hour",
                                       labels={"hour":"Hour of day","Average predicted fare ($)":"Average predicted fare ($)"},
                                       color_discrete_sequence=["#ff7f0e"])
                st.plotly_chart(fig_hour_pred, use_container_width=True)
                st.caption("Without actual fares, we summarize how predictions vary over the day. Peaks may indicate typical surge periods (e.g., commute hours).")

        # By trip distance bucket
        st.markdown("**By trip distance**")
        bins = [0, 1, 3, 5, 10, 20, np.inf]
        labels = ["0–1", "1–3", "3–5", "5–10", "10–20", "20+"]
        outpdf["dist_bin"] = pd.cut(outpdf["trip_distance"], bins=bins, labels=labels, right=False)

        if has_actual:
            dist_mae = (outpdf
                        .groupby("dist_bin")
                        .apply(lambda g: (g["trip_fare"] - g["fare_pred"]).abs().mean())
                        .reset_index(name="MAE"))
            fig_dist_mae = px.bar(dist_mae, x="dist_bin", y="MAE",
                                  title="MAE by Trip Distance Bucket",
                                  labels={"dist_bin":"Trip distance (miles)","MAE":"Mean Absolute Error"},
                                  color_discrete_sequence=["#2ca02c"])
            st.plotly_chart(fig_dist_mae, use_container_width=True)
            st.caption("Average absolute error by distance range. Larger values at the right imply the model underperforms on long trips.")
        else:
            dist_pred = (outpdf.groupby("dist_bin")["fare_pred"].mean()
                                 .reset_index(name="Average predicted fare ($)"))
            fig_dist_pred = px.bar(dist_pred, x="dist_bin", y="Average predicted fare ($)",
                                   title="Average Predicted Fare by Distance",
                                   labels={"dist_bin":"Trip distance (miles)","Average predicted fare ($)":"Average predicted fare ($)"},
                                   color_discrete_sequence=["#d62728"])
            st.plotly_chart(fig_dist_pred, use_container_width=True)
            st.caption("Average predicted fare by trip distance bucket. This shows how the model scales fares with distance.")

        # Residuals histogram (only if actual present)
        if has_actual:
            st.markdown("**Residuals (actual − predicted)**")
            err = outpdf["trip_fare"] - outpdf["fare_pred"]
            fig_hist = px.histogram(x=err, nbins=40,
                                    title="Residuals Distribution (actual − predicted)",
                                    labels={"x":"Residual ($)","y":"Count"})
            fig_hist.update_traces(marker_color="#9467bd")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("Residuals center near zero in a well-calibrated model. Wide or skewed distributions suggest bias or heavy-tail errors.")

        # Scatter: distance vs fares (sample for speed)
        st.markdown("**Fare vs. distance (sample)**")
        sample = outpdf.sample(n=min(3000, len(outpdf)), random_state=7)
        colA, colB = st.columns(2)
        fig_pred_scatter = px.scatter(sample, x="trip_distance", y="fare_pred",
                                      title="Predicted Fare vs. Distance (sample)",
                                      labels={"trip_distance":"Trip distance (miles)","fare_pred":"Predicted fare ($)"},
                                      color_discrete_sequence=["#17becf"])
        colA.plotly_chart(fig_pred_scatter, use_container_width=True)

        if has_actual and "trip_fare" in sample.columns:
            fig_act_scatter = px.scatter(sample, x="trip_distance", y="trip_fare",
                                         title="Actual Fare vs. Distance (sample)",
                                         labels={"trip_distance":"Trip distance (miles)","trip_fare":"Actual fare ($)"},
                                         color_discrete_sequence=["#8c564b"])
            colB.plotly_chart(fig_act_scatter, use_container_width=True)
            st.caption("Comparing actual vs predicted trends against distance helps spot under/over-prediction at long or short trips.")
        else:
            colB.write("Upload with a `trip_fare` column to compare actual fares.")

        # Top absolute errors table (only if actual fares are provided)
        if has_actual and "trip_fare" in outpdf.columns:
            st.subheader("Top 20 absolute errors")
            worst = (outpdf.assign(abs_error=(outpdf["trip_fare"] - outpdf["fare_pred"]).abs())
                             .sort_values("abs_error", ascending=False)
                             .head(20))
            cols_order = [c for c in [
                "trip_distance","trip_minutes","passenger_count","hour","is_weekend",
                "trip_fare","fare_pred","abs_error"
            ] if c in worst.columns]
            st.dataframe(worst[cols_order], use_container_width=True)
            st.caption("Largest gaps between actual and predicted fare. Use these rows to inspect edge cases (e.g., very long trips, late-night pickups).")

        # Model coefficients (Linear Regression)
        with st.expander("Model coefficients"):
            try:
                coef_list = list(model.coefficients)
                # Align length with FEATS just in case
                k = min(len(coef_list), len(FEATS))
                coef_df = pd.DataFrame({
                    "feature": FEATS[:k],
                    "coefficient": coef_list[:k]
                })
                st.dataframe(coef_df, use_container_width=True)
                st.caption("Coefficients are for the **scaled** features the model was trained on. Positive values increase predicted fare; negative values decrease it.")
                st.write(f"Intercept: {getattr(model, 'intercept', float('nan')):.4f}")
            except Exception as _e:
                st.info(f"Could not display coefficients: {_e}")
    except Exception as e:
        st.error(f"Scoring failed: {e}")

st.caption("Tip: For very large files, use the batch scorer script for speed.")