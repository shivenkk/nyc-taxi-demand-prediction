import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import load_model
from scipy import stats

# ------------------------------
# Feature preparation
# ------------------------------
def prepare_features(df: pd.DataFrame) -> tuple:
    exclude_cols = ['pickup_count', 'PULocationID', 'pickup_hour']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['pickup_count'].copy()

    return X, y

# ------------------------------
# Basic metrics
# ------------------------------
def evaluate_basic_metrics(y_true, y_pred, split_name='Test'):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    metrics = {
        'split': split_name,
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'n_samples': len(y_true)
    }

    return metrics

# ------------------------------
# Peak hours evaluation
# ------------------------------
def evaluate_peak_hours(y_true, y_pred, df, split_name='Test'):
    if 'is_rush_hour' not in df.columns:
        print(f"Warning: 'is_rush_hour' not found in {split_name} set")
        return None

    is_peak = df['is_rush_hour'] == 1
    if is_peak.sum() == 0:
        return None

    metrics = evaluate_basic_metrics(
        y_true[is_peak], y_pred[is_peak], split_name=f'{split_name}_peak'
    )
    return metrics

# ------------------------------
# Zone demand evaluation
# ------------------------------
def evaluate_by_zone_type(y_true, y_pred, df, split_name='Test'):
    if 'zone_avg_demand' not in df.columns:
        print(f"Warning: 'zone_avg_demand' not found")
        return []

    demand_quantiles = df['zone_avg_demand'].quantile([0.33, 0.67])
    results = []

    low_mask = df['zone_avg_demand'] <= demand_quantiles[0.33]
    if low_mask.sum() > 0:
        metrics = evaluate_basic_metrics(y_true[low_mask], y_pred[low_mask], f'{split_name}_low_demand')
        results.append(metrics)

    med_mask = (df['zone_avg_demand'] > demand_quantiles[0.33]) & \
               (df['zone_avg_demand'] <= demand_quantiles[0.67])
    if med_mask.sum() > 0:
        metrics = evaluate_basic_metrics(y_true[med_mask], y_pred[med_mask], f'{split_name}_medium_demand')
        results.append(metrics)

    high_mask = df['zone_avg_demand'] > demand_quantiles[0.67]
    if high_mask.sum() > 0:
        metrics = evaluate_basic_metrics(y_true[high_mask], y_pred[high_mask], f'{split_name}_high_demand')
        results.append(metrics)

    return results

# ------------------------------
# Plots
# ------------------------------
def plot_predictions(y_true, y_pred, split_name, output_path):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Pickup Count')
    axes[0].set_ylabel('Predicted Pickup Count')
    axes[0].set_title(f'Predicted vs Actual ({split_name})')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Pickup Count')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title(f'Residuals ({split_name})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Prediction plot saved: {output_path}")

def plot_error_distribution(y_true, y_pred, split_name, output_path):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    errors = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Error Distribution ({split_name})')
    axes[0].grid(True, alpha=0.3)

    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot ({split_name})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Error distribution plot saved: {output_path}")

# ------------------------------
# Full evaluation
# ------------------------------
def evaluate_model(model_path: str, output_dir: str = None):
    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL: {model_path.name}")
    print(f"{'='*70}\n")

    # Load scaler and model
    scaler_X = joblib.load(model_path.parent / 'scaler.pkl')  # Features
    scaler_y = joblib.load(model_path.parent / 'scaler_y.pkl') if (model_path.parent / 'scaler_y.pkl').exists() else None
    model = load_model(model_path)

    # Load datasets
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')

    # Prepare features and targets
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    # Scale features
    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Predict
    y_train_pred = model.predict(X_train_scaled).flatten()
    y_val_pred = model.predict(X_val_scaled).flatten()
    y_test_pred = model.predict(X_test_scaled).flatten()

    # Inverse transform y if scaler_y exists
    if scaler_y:
        y_train = scaler_y.inverse_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val = scaler_y.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
        y_test = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
        y_train_pred = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        y_val_pred = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
        y_test_pred = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # --- Metrics ---
    all_metrics = []
    for split_name, y_true, y_pred in [
        ('Train', y_train, y_train_pred),
        ('Validation', y_val, y_val_pred),
        ('Test', y_test, y_test_pred)
    ]:
        metrics = evaluate_basic_metrics(y_true, y_pred, split_name)
        all_metrics.append(metrics)
        print(f" {split_name} - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.2f}")

    # Peak hour evaluation
    print("\nPeak hours evaluation:")
    for split_name, df, y_pred in [
        ('Train', train_df, y_train_pred),
        ('Validation', val_df, y_val_pred),
        ('Test', test_df, y_test_pred)
    ]:
        metrics = evaluate_peak_hours(df['pickup_count'], y_pred, df, split_name)
        if metrics:
            all_metrics.append(metrics)
            print(f" {split_name} Peak - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")

    # Zone type evaluation (test set only)
    print("\nZone demand evaluation (Test set):")
    zone_metrics = evaluate_by_zone_type(y_test, y_test_pred, test_df, 'Test')
    all_metrics.extend(zone_metrics)
    for m in zone_metrics:
        print(f" {m['split']} - RMSE: {m['rmse']:.2f}, R²: {m['r2']:.4f}")

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / 'evaluation_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics saved: {metrics_path}")

    # Visualizations
    plot_predictions(y_test, y_test_pred, 'Test', output_dir / 'predictions_plot.png')
    plot_error_distribution(y_test, y_test_pred, 'Test', output_dir / 'error_distribution.png')

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE ✅")
    print(f"{'='*70}")

    return metrics_df

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/dnn/model.keras'
    evaluate_model(model_path)
