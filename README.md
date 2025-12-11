# NYC Taxi Demand Prediction

Predicting hourly taxi pickup demand across NYC zones using the TLC Yellow Taxi dataset (January 2024). We compare Linear Regression, Random Forest, XGBoost, DNN, and LSTM models.

## Setup

```bash
git clone https://github.com/shivenkk/nyc-taxi-demand-prediction.git
cd nyc-taxi-demand-prediction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download January 2024 Yellow Taxi data:
```bash
mkdir -p data/raw
curl -o data/raw/yellow_tripdata_2024-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```

## Running the Pipeline

### 1. Preprocess data
```bash
python src/data/preprocess_pipeline.py
```

### 2. Feature engineering
```bash
python src/features/build_features.py
```

### 3. Train/val/test split
```bash
python src/features/split_data.py
```

### 4. Train models
```bash
# Baseline (Linear Regression)
python src/models/baseline_model.py

# Random Forest
python src/models/random_forest_model.py

# XGBoost
python src/models/xgboost_model.py

# DNN
python src/models/dnn_model.py

# LSTM
python src/models/lstm_model.py
```

### 5. Evaluate models
```bash
python src/evaluation/evaluate_model.py models/baseline/model.pkl "Linear Regression"
python src/evaluation/evaluate_model.py models/xgboost/model.pkl "XGBoost"
python src/evaluation/evaluate_model.py models/lstm/model.pkl "LSTM"
```

## Project Structure

```
├── data/
│   ├── raw/                 # Raw parquet files
│   └── processed/           # Processed features and splits
├── models/                  # Saved models and evaluation results
│   ├── baseline/
│   ├── random_forest/
│   ├── xgboost/
│   ├── dnn/
│   └── lstm/
├── src/
│   ├── data/                # Data loading and cleaning
│   ├── features/            # Feature engineering
│   ├── models/              # Model training scripts
│   └── evaluation/          # Evaluation utilities
└── report/                  # Final report
```

## Results

| Model | Test RMSE | Test R² |
|-------|-----------|---------|
| Linear Regression | 17.38 | 0.947 |
| Random Forest | -- | -- |
| **XGBoost** | **15.50** | **0.958** |
| DNN | -- | -- |
| LSTM | 33.99 | 0.798 |

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn
- xgboost
- torch (PyTorch)
- matplotlib, seaborn
