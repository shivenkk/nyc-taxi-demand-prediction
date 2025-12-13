import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('src/models')
from lstm_utils import LSTMWrapper, LSTMNet
from dnn_model import DNNWrapper, DNNNet

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target (same as training)"""
    exclude_cols = ['pickup_count', 'PULocationID', 'pickup_hour']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['pickup_count'].copy()
    
    return X, y

def evaluate_basic_metrics(model, X, y, split_name='Test'):
    """Calculate basic regression metrics"""
    y_pred = model.predict(X)
    
    metrics = {
        'split': split_name,
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred),
        'mae': mean_absolute_error(y, y_pred),
        'n_samples': len(y)
    }
    
    return metrics, y_pred

def evaluate_peak_hours(model, X, y, df, split_name='Test'):
    """Evaluate performance during peak hours only"""
    if 'is_rush_hour' not in df.columns:
        print(f"Warning: 'is_rush_hour' not found in {split_name} set")
        return None
    
    is_peak = df['is_rush_hour'] == 1
    
    if is_peak.sum() == 0:
        return None
    
    X_peak = X[is_peak]
    y_peak = y[is_peak]
    y_pred_peak = model.predict(X_peak)
    
    return {
        'split': f'{split_name}_peak',
        'rmse': np.sqrt(mean_squared_error(y_peak, y_pred_peak)),
        'r2': r2_score(y_peak, y_pred_peak),
        'mae': mean_absolute_error(y_peak, y_pred_peak),
        'n_samples': len(y_peak)
    }

def evaluate_by_zone_type(model, X, y, df, split_name='Test'):
    """
    Evaluate by zone demand level (high/medium/low based on zone_avg_demand)
    """
    if 'zone_avg_demand' not in df.columns:
        print(f"Warning: 'zone_avg_demand' not found")
        return []
    
    # Categorize zones by demand level
    demand_quantiles = df['zone_avg_demand'].quantile([0.33, 0.67])
    
    results = []
    
    # Low demand zones
    low_mask = df['zone_avg_demand'] <= demand_quantiles[0.33]
    if low_mask.sum() > 0:
        metrics, _ = evaluate_basic_metrics(model, X[low_mask], y[low_mask], 
                                           f'{split_name}_low_demand')
        results.append(metrics)
    
    # Medium demand zones
    med_mask = (df['zone_avg_demand'] > demand_quantiles[0.33]) & \
               (df['zone_avg_demand'] <= demand_quantiles[0.67])
    if med_mask.sum() > 0:
        metrics, _ = evaluate_basic_metrics(model, X[med_mask], y[med_mask],
                                           f'{split_name}_medium_demand')
        results.append(metrics)
    
    # High demand zones
    high_mask = df['zone_avg_demand'] > demand_quantiles[0.67]
    if high_mask.sum() > 0:
        metrics, _ = evaluate_basic_metrics(model, X[high_mask], y[high_mask],
                                           f'{split_name}_high_demand')
        results.append(metrics)
    
    return results

def plot_predictions(y_true, y_pred, split_name, output_path):
    """Create prediction vs actual and residual plots"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Pickup Count')
    axes[0].set_ylabel('Predicted Pickup Count')
    axes[0].set_title(f'Predicted vs Actual ({split_name})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Pickup Count')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title(f'Residual Plot ({split_name})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Plot saved to: {output_path}")

def plot_error_distribution(y_true, y_pred, split_name, output_path):
    """Plot error distribution"""
    errors = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of errors
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Prediction Error (Actual - Predicted)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Error Distribution ({split_name})')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot ({split_name})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Error distribution plot saved to: {output_path}")

def evaluate_model(model_path: str, model_name: str = 'Model', 
                  output_dir: str = None):
    """
    Complete evaluation pipeline for a trained model
    
    Args:
        model_path: Path to saved model (.pkl file)
        model_name: Name of the model (for outputs)
        output_dir: Where to save evaluation results (default: same as model)
    """
    model_path = Path(model_path)
    
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"EVALUATING: {model_name}")
    
    # Load model
    print("Loading model...")
    model = joblib.load(model_path)
    print(f" Model loaded from: {model_path}")
    
    # Load data
    print("\nLoading data")
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')

    # Sort for LSTM (needs sequential data)
    if 'LSTM' in model_name:
        train_df = train_df.sort_values(['PULocationID', 'pickup_hour']).reset_index(drop=True)
        val_df = val_df.sort_values(['PULocationID', 'pickup_hour']).reset_index(drop=True)
        test_df = test_df.sort_values(['PULocationID', 'pickup_hour']).reset_index(drop=True)
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")
    
    # Prepare features
    print("\nPreparing features")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # Evaluate on all splits
    print("\nEvaluating on all splits...")
    all_metrics = []
    
    # Train set
    metrics, y_train_pred = evaluate_basic_metrics(model, X_train, y_train, 'Train')
    all_metrics.append(metrics)
    print(f"  Train - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
    
    # Validation set
    metrics, y_val_pred = evaluate_basic_metrics(model, X_val, y_val, 'Validation')
    all_metrics.append(metrics)
    print(f"  Val   - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
    
    # Test set
    metrics, y_test_pred = evaluate_basic_metrics(model, X_test, y_test, 'Test')
    all_metrics.append(metrics)
    print(f"  Test  - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
    
    # Peak hours evaluation
    print("\nEvaluating peak hours...")
    for split_name, X, y, df in [('Train', X_train, y_train, train_df),
                                  ('Val', X_val, y_val, val_df),
                                  ('Test', X_test, y_test, test_df)]:
        peak_metrics = evaluate_peak_hours(model, X, y, df, split_name)
        if peak_metrics:
            all_metrics.append(peak_metrics)
            print(f"  {split_name} Peak - RMSE: {peak_metrics['rmse']:.2f}")
    
    # Zone-type evaluation (test set only)
    print("\nEvaluating by zone demand level (test set)...")
    zone_metrics = evaluate_by_zone_type(model, X_test, y_test, test_df, 'Test')
    all_metrics.extend(zone_metrics)
    for m in zone_metrics:
        print(f"  {m['split']} - RMSE: {m['rmse']:.2f}, R²: {m['r2']:.4f}")
    
    # Save metrics
    print("\nSaving results...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / 'evaluation_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f" Metrics saved to: {metrics_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_predictions(y_test, y_test_pred, 'Test', 
                    output_dir / 'predictions_plot.png')
    plot_error_distribution(y_test, y_test_pred, 'Test',
                          output_dir / 'error_distribution.png')
    
    print("EVALUATION FINISHED")
    
    return metrics_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else 'Model'
    else:
        # Default: evaluate baseline model
        model_path = 'models/baseline/model.pkl'
        model_name = 'Linear Regression Baseline'
    
    evaluate_model(model_path, model_name)
