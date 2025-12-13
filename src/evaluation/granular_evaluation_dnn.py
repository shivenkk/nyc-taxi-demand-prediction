import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')

# ==========================
# Model loading and prediction
# ==========================
def load_any_model(model_path):
    model_path = str(model_path)
    if model_path.endswith(".keras") or model_path.endswith(".h5") or Path(model_path).is_dir():
        print("Detected Keras/TensorFlow model. Loading with keras.models.load_model()...")
        return keras.models.load_model(model_path)
    print("Detected scikit-learn model. Loading with joblib.load()...")
    return joblib.load(model_path)

def predict_model(model, X, y_scaler=None):
    """Unified prediction for sklearn + keras models with optional y inverse scaling"""
    X_array = np.array(X, dtype=float)
    if hasattr(model, "predict") and "keras" in str(type(model)).lower():
        preds = model.predict(X_array, verbose=0)
    else:
        preds = model.predict(X_array)
    preds = np.array(preds).reshape(-1)
    if y_scaler:
        preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    return preds

# ==========================
# Data preparation
# ==========================
def prepare_features(df: pd.DataFrame) -> tuple:
    exclude_cols = ['pickup_count', 'PULocationID', 'pickup_hour']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy().astype(float)
    y = df['pickup_count'].copy()
    return X, y

# ==========================
# Helper metrics
# ==========================
def calculate_accuracy_metrics(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    acc_5 = (errors <= 5).mean() * 100
    acc_10 = (errors <= 10).mean() * 100
    acc_15 = (errors <= 15).mean() * 100
    percentage_errors = np.abs((y_true - y_pred) / (y_true + 1)) * 100
    acc_10pct = (percentage_errors <= 10).mean() * 100
    acc_20pct = (percentage_errors <= 20).mean() * 100
    acc_30pct = (percentage_errors <= 30).mean() * 100
    mape = percentage_errors.mean()
    return {
        'acc_within_5': acc_5,
        'acc_within_10': acc_10,
        'acc_within_15': acc_15,
        'acc_within_10pct': acc_10pct,
        'acc_within_20pct': acc_20pct,
        'acc_within_30pct': acc_30pct,
        'mape': mape
    }

# ==========================
# Overall evaluation
# ==========================
def evaluate_overall_performance(model, train_df, test_df, output_dir, y_scaler=None):
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    y_train_pred = predict_model(model, X_train, y_scaler)
    y_test_pred = predict_model(model, X_test, y_scaler)

    train_acc = calculate_accuracy_metrics(y_train, y_train_pred)
    test_acc = calculate_accuracy_metrics(y_test, y_test_pred)

    train_metrics = {
        'split': 'Training',
        'n_samples': len(y_train),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred),
        'mean_actual': y_train.mean(),
        'std_actual': y_train.std(),
        'mean_predicted': y_train_pred.mean(),
        'std_predicted': y_train_pred.std(),
        **train_acc
    }
    test_metrics = {
        'split': 'Test',
        'n_samples': len(y_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred),
        'mean_actual': y_test.mean(),
        'std_actual': y_test.std(),
        'mean_predicted': y_test_pred.mean(),
        'std_predicted': y_test_pred.std(),
        **test_acc
    }

    overall_df = pd.DataFrame([train_metrics, test_metrics])
    overall_df.to_csv(output_dir / 'overall_performance.csv', index=False)
    print("✓ Overall performance metrics saved")
    return train_metrics, test_metrics, y_test_pred

# ==========================
# Hourly evaluation
# ==========================
def evaluate_by_hour(y_true, y_pred, df, output_dir):
    results = []
    for hour in range(24):
        mask = df['hour'] == hour
        if mask.sum() > 0:
            y_h = y_true[mask]
            pred_h = y_pred[mask]
            results.append({
                'hour': hour,
                'rmse': np.sqrt(mean_squared_error(y_h, pred_h)),
                'mae': mean_absolute_error(y_h, pred_h),
                'r2': r2_score(y_h, pred_h),
                'n_samples': mask.sum(),
                'mean_actual': y_h.mean(),
                'mean_predicted': pred_h.mean()
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_by_hour.csv', index=False)
    print("✓ Hourly analysis saved")
    return results_df

# ==========================
# Day-of-week evaluation
# ==========================
def evaluate_by_day_of_week(y_true, y_pred, df, output_dir):
    results = []
    day_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for dow in range(7):
        mask = df['day_of_week'] == dow
        if mask.sum() > 0:
            y_d = y_true[mask]
            pred_d = y_pred[mask]
            results.append({
                'day_of_week': dow,
                'day_name': day_names[dow],
                'rmse': np.sqrt(mean_squared_error(y_d, pred_d)),
                'mae': mean_absolute_error(y_d, pred_d),
                'r2': r2_score(y_d, pred_d),
                'n_samples': mask.sum(),
                'mean_actual': y_d.mean(),
                'mean_predicted': pred_d.mean()
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_by_day_of_week.csv', index=False)
    print("✓ Day-of-week analysis saved")
    return results_df

# ==========================
# Peak vs Off-peak
# ==========================
def evaluate_peak_vs_offpeak(y_true, y_pred, df, output_dir):
    results = []
    peak_mask = df['is_rush_hour'] == 1
    offpeak_mask = df['is_rush_hour'] == 0
    for label, mask in zip(['Peak Hours','Off-Peak Hours'], [peak_mask, offpeak_mask]):
        if mask.sum() > 0:
            y_sub = y_true[mask]
            pred_sub = y_pred[mask]
            results.append({
                'period': label,
                'rmse': np.sqrt(mean_squared_error(y_sub, pred_sub)),
                'mae': mean_absolute_error(y_sub, pred_sub),
                'r2': r2_score(y_sub, pred_sub),
                'n_samples': mask.sum(),
                'mean_actual': y_sub.mean(),
                'mean_predicted': pred_sub.mean()
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_peak_vs_offpeak.csv', index=False)
    print("✓ Peak vs off-peak analysis saved")
    return results_df

# ==========================
# Top zones evaluation
# ==========================
def evaluate_top_zones(y_true, y_pred, df, output_dir, top_n=10):
    df_eval = df.copy()
    df_eval['error'] = np.abs(y_true - y_pred)
    zone_counts = df_eval.groupby('PULocationID')['pickup_count'].sum()
    top_zones = zone_counts.sort_values(ascending=False).head(top_n).index
    results = []
    for zone in top_zones:
        mask = df_eval['PULocationID'] == zone
        y_zone = y_true[mask]
        pred_zone = y_pred[mask]
        results.append({
            'zone': zone,
            'rmse': np.sqrt(mean_squared_error(y_zone, pred_zone)),
            'mae': mean_absolute_error(y_zone, pred_zone),
            'r2': r2_score(y_zone, pred_zone),
            'n_samples': mask.sum(),
            'mean_actual': y_zone.mean(),
            'mean_predicted': pred_zone.mean()
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_top_zones.csv', index=False)
    print("✓ Top zones evaluation saved")
    return results_df

# ==========================
# Zone demand level evaluation
# ==========================
def evaluate_by_zone_demand_level(y_true, y_pred, df, output_dir):
    df_eval = df.copy()
    df_eval['error'] = np.abs(y_true - y_pred)
    zone_mean = df_eval.groupby('PULocationID')['pickup_count'].mean()
    labels = pd.qcut(zone_mean, q=3, labels=['Low','Medium','High'])
    df_eval['demand_level'] = df_eval['PULocationID'].map(labels)
    results = []
    for level in ['Low','Medium','High']:
        mask = df_eval['demand_level'] == level
        if mask.sum() > 0:
            y_sub = y_true[mask]
            pred_sub = y_pred[mask]
            results.append({
                'demand_level': level,
                'rmse': np.sqrt(mean_squared_error(y_sub, pred_sub)),
                'mae': mean_absolute_error(y_sub, pred_sub),
                'r2': r2_score(y_sub, pred_sub),
                'n_samples': mask.sum(),
                'mean_actual': y_sub.mean(),
                'mean_predicted': pred_sub.mean()
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_by_zone_demand_level.csv', index=False)
    print("✓ Zone demand level analysis saved")
    return results_df

# ==========================
# Demand magnitude evaluation
# ==========================
def evaluate_by_demand_magnitude(y_true, y_pred, df, output_dir):
    df_eval = df.copy()
    df_eval['error'] = np.abs(y_true - y_pred)
    df_eval['demand_magnitude'] = pd.qcut(df_eval['pickup_count'], q=3, labels=['Low','Medium','High'])
    results = []
    for level in ['Low','Medium','High']:
        mask = df_eval['demand_magnitude'] == level
        if mask.sum() > 0:
            y_sub = y_true[mask]
            pred_sub = y_pred[mask]
            results.append({
                'demand_magnitude': level,
                'rmse': np.sqrt(mean_squared_error(y_sub, pred_sub)),
                'mae': mean_absolute_error(y_sub, pred_sub),
                'r2': r2_score(y_sub, pred_sub),
                'n_samples': mask.sum(),
                'mean_actual': y_sub.mean(),
                'mean_predicted': pred_sub.mean()
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_by_demand_magnitude.csv', index=False)
    print("✓ Demand magnitude analysis saved")
    return results_df

# ==========================
# Error heatmap
# ==========================
def plot_error_heatmap(y_true, y_pred, df, output_dir):
    df_eval = df.copy()
    df_eval['error'] = np.abs(y_true - y_pred)
    error_pivot = df_eval.groupby(['day_of_week','hour'])['error'].mean().reset_index()
    error_matrix = error_pivot.pivot(index='day_of_week', columns='hour', values='error')
    fig, ax = plt.subplots(figsize=(16,6))
    sns.heatmap(error_matrix, cmap='YlOrRd', ax=ax, cbar_kws={'label':'Mean Absolute Error'})
    ax.set_xlabel('Hour')
    ax.set_ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Error heatmap saved")
