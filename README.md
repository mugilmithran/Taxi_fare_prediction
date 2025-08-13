## NYC Taxi Fare Prediction — PySpark (Bronze/Silver/Gold + MLlib + Streamlit)

End-to-end Spark project that ingests NYC Taxi data (≥1 GB), curates Delta Lake tables (Bronze → Silver → Gold), builds features, trains MLlib models (Linear Regression vs Decision Tree), batch-scores predictions, and ships a Streamlit app for upload-and-predict with clear visuals.

⸻

# What this project does 
	•	Ingest monthly NYC Yellow Taxi Parquet files into a Bronze Delta table.
	•	Clean & type into Silver (drop duplicates, filter invalid rows).
	•	Aggregate & feature-engineer into Gold (hourly stats + ML features).
	•	Train models in Spark MLlib (we choose Linear Regression for the final app).
	•	Batch-score predictions to a Delta table.
	•	Frontend: a Streamlit page where anyone uploads a CSV and gets fare predictions + simple charts (hour, distance buckets, residuals).

⸻

# Repo structure

<img width="704" height="263" alt="image" src="https://github.com/user-attachments/assets/446df752-da23-4460-bbd8-4ed570d4012d" />


Prereqs

	•	Python 3.9.x
 
	•	Java 17
 
	•	Spark 3.5.1 (PySpark installed via pip)
 
	•	Delta Lake 3.2.0 (via delta-spark)

# Create the env (example):

conda create -n spark39 python=3.9 -y
conda activate spark39
pip install pyspark==3.5.1 delta-spark==3.2.0 pyarrow pandas plotly streamlit

Quickstart (end-to-end)

All commands from the repo root.

1) Download ≥1 GB of NYC Taxi Parquet (example months)

Put files under data/nyc/ (e.g., yellow_tripdata_2022-01.parquet, …).
Tip: 12+ monthly Parquet files usually exceeds 1 GB.

2) Ingest → Bronze (Delta)
python src/ingest/nyc_ingest_parquet.py
 writes: data/bronze/nyc_taxi  (Delta)

3) Clean & Explore → Silver/Gold
python src/transform/curations.py
 writes: data/silver/trips_clean (Delta)
         data/gold/hourly_stats  (Parquet/Delta depending on script)

4) Build features
python src/features/build_features_v2.py
# writes: data/gold/features_reg_v2 (Parquet)

5) Train models (LR + DT) and save scaler + metrics
python src/ml/train_lr_dt_v2.py
 saves:
   artifacts/preprocessors/lin_scaler
   artifacts/models/linreg_model
   artifacts/models/dt_model
   artifacts/metrics/eval_lr_v2/overall.json  (train/test RMSE, MAE, R²)

6) Streamlit UI (upload-and-predict)
streamlit run app/streamlit_app.py

Open the URL shown (usually http://localhost:8501).

Streamlit — how to use the uploader

Accepted schemas (either one):

A) Timestamp route
	•	trip_distance (+ optional passenger_count)
	•	tpep_pickup_datetime or pickup_datetime (optional tpep_dropoff_datetime/dropoff_datetime for better trip_minutes)

B) Feature route
	•	trip_distance, trip_minutes, passenger_count, hour, is_weekend

Optional: include trip_fare to see errors (MAE, residuals) and the Top 20 worst errors table.

What the visuals show (in the app):
	•	MAE by hour – when actual fares are present; highlights hours where the model struggles.
	•	Average predicted fare by hour – when actual isn’t present; shows daily patterns.
	•	MAE by distance bucket – checks short vs long trips.
	•	Residuals histogram – distribution of (actual − predicted); should center near 0.
	•	Scatter (sampled) – fare vs distance for predicted and (if present) actual.
	•	Model coefficients – LR coefficients for interpretability (scaled features).

Typical metrics (your numbers will vary)

After cleaning outliers and using v2 features:
	•	Linear Regression (test): RMSE ≈ 3.1, MAE ≈ 0.95, R² ≈ 0.93
	•	Decision Tree (test): similar but slightly worse generalization than LR

Final model in the app: Linear Regression (fast, stable, interpretable).

All metrics are saved to artifacts/metrics/eval_lr_v2/overall.json.

Why the Bronze/Silver/Gold pattern?
	•	Bronze: raw data, append-only, reproducible ingest.
	•	Silver: cleaned & typed tables for analytics.
	•	Gold: feature tables and marts tuned for ML/dashboards.

It makes your pipeline easy to reason about and simple to re-run.

Commands cheat-sheet

# 0) Env
conda activate spark39

# 1) Ingest
python src/ingest/nyc_ingest_parquet.py

# 2) Curate
python src/transform/curations.py

# 3) Features
python src/features/build_features_v2.py

# 4) Train
python src/ml/train_lr_dt_v2.py

# 5) Batch predict (optional)
python src/serve/batch_score_lr.py

# 6) Sanity checks
python src/qa/sanity_checks.py
python src/qa/sanity_checks_pred.py

# 7) Quick report (PNG charts)
python src/report/quick_report.py

# 8) Frontend
streamlit run app/streamlit_app.py
