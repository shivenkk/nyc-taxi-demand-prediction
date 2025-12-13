#!/usr/bin/env bash

set -e  # Exit immediately if any command fails

echo "=== Starting NYC Taxi ML Pipeline ==="

# -----------------------------
# 1. Download data
# -----------------------------
echo "Downloading January 2024 Yellow Taxi data..."
mkdir -p data/raw

curl -o data/raw/yellow_tripdata_2024-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet

# -----------------------------
# 2. Preprocess data
# -----------------------------
echo "Running preprocessing..."
python src/data/preprocess_pipeline.py

# -----------------------------
# 3. Feature engineering
# -----------------------------
echo "Building features..."
python src/features/build_features.py

# -----------------------------
# 4. Train/val/test split
# -----------------------------
echo "Splitting data..."
python src/features/split_data.py

# -----------------------------
# 5. Train models
# -----------------------------
echo "Training baseline (Linear Regression)..."
python src/models/baseline_model.py

echo "Training Random Forest..."
python src/models/random_forest_model.py

echo "Training XGBoost..."
python src/models/xgboost_model.py

echo "Training DNN..."
python src/models/dnn_model.py

echo "Training LSTM..."
python src/models/lstm_model.py

# -----------------------------
# 6. Evaluate models
# -----------------------------
echo "Evaluating models..."

python src/evaluation/evaluate_model.py models/baseline/model.pkl "Linear Regression"
python src/evaluation/evaluate_model.py models/random_forest/model.pkl "Random Forest"
python src/evaluation/evaluate_model.py models/xgboost/model.pkl "XGBoost"
python src/evaluation/evaluate_model.py models/dnn/model.pkl "DNN"
python src/evaluation/evaluate_model.py models/lstm/model.pkl "LSTM"

echo "=== Pipeline completed successfully ==="
