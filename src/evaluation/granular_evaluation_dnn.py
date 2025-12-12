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

def load_any_model(model_path):
    model_path = str(model_path)

    # If it's a keras model (SavedModel or .keras or .h5)
    if model_path.endswith(".keras") or model_path.endswith(".h5") or Path(model_path).is_dir():
        print("Detected Keras/TensorFlow model. Loading with keras.models.load_model()...")
        return keras.models.load_model(model_path)

    # Otherwise assume sklearn joblib model
    print("Detected scikit-learn model. Loading with joblib.load()...")
    return joblib.load(model_path)

def predict_model(model, X):
    """Unified prediction for sklearn + keras models."""
    # Convert DataFrame → numpy for keras models
    if hasattr(model, "predict") and "keras" in str(type(model)).lower():
        preds = model.predict(np.array(X), verbose=0)
    else:
        preds = model.predict(X)

    # Flatten (handles keras (n,1), sklearn (n,), etc.)
    return np.array(preds).reshape(-1)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target"""
    exclude_cols = ['pickup_count', 'PULocationID', 'pickup_hour']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy().astype(float)
    y = df['pickup_count'].copy()
    
    return X, y

def evaluate_overall_performance(model, train_df, test_df, output_dir):
    """
    Evaluate overall performance on training and test sets
    Print and save summary metrics including accuracy measures
    """
    print(f"\n{'='*70}")
    print("OVERALL MODEL PERFORMANCE")
    print(f"{'='*70}\n")
    
    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)
    
    # Generate predictions
    y_train_pred = predict_model(model, X_train)
    y_test_pred = predict_model(model, X_test)
    
    # Calculate accuracy metrics (for regression)
    # Accuracy within tolerance thresholds
    def calculate_accuracy_metrics(y_true, y_pred):
        """Calculate accuracy at different tolerance levels"""
        errors = np.abs(y_true - y_pred)
        
        # Absolute tolerance accuracy
        acc_5 = (errors <= 5).mean() * 100  # Within 5 pickups
        acc_10 = (errors <= 10).mean() * 100  # Within 10 pickups
        acc_15 = (errors <= 15).mean() * 100  # Within 15 pickups
        
        # Percentage tolerance accuracy (MAPE-based)
        percentage_errors = np.abs((y_true - y_pred) / (y_true + 1)) * 100  # +1 to avoid div by 0
        acc_10pct = (percentage_errors <= 10).mean() * 100  # Within 10%
        acc_20pct = (percentage_errors <= 20).mean() * 100  # Within 20%
        acc_30pct = (percentage_errors <= 30).mean() * 100  # Within 30%
        
        # Mean Absolute Percentage Error
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
    
    train_acc = calculate_accuracy_metrics(y_train, y_train_pred)
    test_acc = calculate_accuracy_metrics(y_test, y_test_pred)
    
    # Calculate standard metrics
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
    
    # Print results
    print("Training Set Performance:")
    print(f"  Samples:        {train_metrics['n_samples']:>10,}")
    print(f"  RMSE:           {train_metrics['rmse']:>10.2f}")
    print(f"  MAE:            {train_metrics['mae']:>10.2f}")
    print(f"  R²:             {train_metrics['r2']:>10.4f}")
    print(f"  MAPE:           {train_metrics['mape']:>10.2f}%")
    print(f"  Mean (Actual):  {train_metrics['mean_actual']:>10.2f}")
    print(f"  Mean (Pred):    {train_metrics['mean_predicted']:>10.2f}")
    
    print("\n  Accuracy Metrics:")
    print(f"    Within ±5 pickups:     {train_metrics['acc_within_5']:>6.2f}%")
    print(f"    Within ±10 pickups:    {train_metrics['acc_within_10']:>6.2f}%")
    print(f"    Within ±15 pickups:    {train_metrics['acc_within_15']:>6.2f}%")
    print(f"    Within ±10% error:     {train_metrics['acc_within_10pct']:>6.2f}%")
    print(f"    Within ±20% error:     {train_metrics['acc_within_20pct']:>6.2f}%")
    print(f"    Within ±30% error:     {train_metrics['acc_within_30pct']:>6.2f}%")
    
    print("\nTest Set Performance:")
    print(f"  Samples:        {test_metrics['n_samples']:>10,}")
    print(f"  RMSE:           {test_metrics['rmse']:>10.2f}")
    print(f"  MAE:            {test_metrics['mae']:>10.2f}")
    print(f"  R²:             {test_metrics['r2']:>10.4f}")
    print(f"  MAPE:           {test_metrics['mape']:>10.2f}%")
    print(f"  Mean (Actual):  {test_metrics['mean_actual']:>10.2f}")
    print(f"  Mean (Pred):    {test_metrics['mean_predicted']:>10.2f}")
    
    print("\n  Accuracy Metrics:")
    print(f"    Within ±5 pickups:     {test_metrics['acc_within_5']:>6.2f}%")
    print(f"    Within ±10 pickups:    {test_metrics['acc_within_10']:>6.2f}%")
    print(f"    Within ±15 pickups:    {test_metrics['acc_within_15']:>6.2f}%")
    print(f"    Within ±10% error:     {test_metrics['acc_within_10pct']:>6.2f}%")
    print(f"    Within ±20% error:     {test_metrics['acc_within_20pct']:>6.2f}%")
    print(f"    Within ±30% error:     {test_metrics['acc_within_30pct']:>6.2f}%")
    
    # Calculate overfitting metrics
    rmse_diff = train_metrics['rmse'] - test_metrics['rmse']
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    acc_diff = train_metrics['acc_within_10'] - test_metrics['acc_within_10']
    
    print("\nGeneralization Analysis:")
    print(f"  RMSE difference (Train - Test):    {rmse_diff:>10.2f}")
    print(f"  R² difference (Train - Test):      {r2_diff:>10.4f}")
    print(f"  Acc±10 difference (Train - Test):  {acc_diff:>10.2f}%")
    
    if abs(rmse_diff) < 2 and abs(r2_diff) < 0.05:
        print(f"  Status: ✓ Excellent generalization")
    elif rmse_diff < -5:
        print(f"  Status: ⚠ Potential underfitting")
    elif rmse_diff > 5:
        print(f"  Status: ⚠ Potential overfitting")
    else:
        print(f"  Status: ✓ Good generalization")
    
    print(f"{'='*70}\n")
    
    # Save metrics
    overall_df = pd.DataFrame([train_metrics, test_metrics])
    overall_df.to_csv(output_dir / 'overall_performance.csv', index=False)
    
    # Create enhanced comparison plots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Traditional metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Accuracy metrics
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])
    
    colors = ['#e74c3c', '#3498db']
    
    # Plot 1: RMSE
    rmse_values = [train_metrics['rmse'], test_metrics['rmse']]
    ax1.bar(['Training', 'Test'], rmse_values, color=colors, alpha=0.7)
    ax1.set_ylabel('RMSE')
    ax1.set_title('Root Mean Squared Error')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmse_values):
        ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Plot 2: R²
    r2_values = [train_metrics['r2'], test_metrics['r2']]
    ax2.bar(['Training', 'Test'], r2_values, color=colors, alpha=0.7)
    ax2.set_ylabel('R²')
    ax2.set_title('R² Score')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(r2_values):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Plot 3: MAPE
    mape_values = [train_metrics['mape'], test_metrics['mape']]
    ax3.bar(['Training', 'Test'], mape_values, color=colors, alpha=0.7)
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('Mean Absolute Percentage Error')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mape_values):
        ax3.text(i, v, f'{v:.2f}%', ha='center', va='bottom')
    
    # Plot 4: Accuracy metrics comparison
    accuracy_types = ['±5 pickups', '±10 pickups', '±15 pickups', 
                      '±10% error', '±20% error', '±30% error']
    train_accs = [train_metrics['acc_within_5'], train_metrics['acc_within_10'], 
                  train_metrics['acc_within_15'], train_metrics['acc_within_10pct'],
                  train_metrics['acc_within_20pct'], train_metrics['acc_within_30pct']]
    test_accs = [test_metrics['acc_within_5'], test_metrics['acc_within_10'],
                 test_metrics['acc_within_15'], test_metrics['acc_within_10pct'],
                 test_metrics['acc_within_20pct'], test_metrics['acc_within_30pct']]
    
    x = np.arange(len(accuracy_types))
    width = 0.35
    
    ax4.bar(x - width/2, train_accs, width, label='Training', color=colors[0], alpha=0.7)
    ax4.bar(x + width/2, test_accs, width, label='Test', color=colors[1], alpha=0.7)
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Prediction Accuracy at Different Tolerance Levels')
    ax4.set_xticks(x)
    ax4.set_xticklabels(accuracy_types, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 100])
    
    # Plot 5: MAE
    mae_values = [train_metrics['mae'], test_metrics['mae']]
    ax5.bar(['Training', 'Test'], mae_values, color=colors, alpha=0.7)
    ax5.set_ylabel('MAE')
    ax5.set_title('Mean Absolute Error')
    ax5.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_values):
        ax5.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.savefig(output_dir / 'overall_performance_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Overall performance metrics saved\n")
    
    return train_metrics, test_metrics, y_test_pred

def evaluate_by_hour(y_true, y_pred, df, output_dir):
    """Evaluate performance by hour of day"""
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
    
    # Save
    results_df.to_csv(output_dir / 'metrics_by_hour.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE by hour
    axes[0, 0].bar(results_df['hour'], results_df['rmse'], alpha=0.7, color='steelblue')
    axes[0, 0].axhline(y=results_df['rmse'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["rmse"].mean():.2f}')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE by Hour of Day')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² by hour
    axes[0, 1].bar(results_df['hour'], results_df['r2'], alpha=0.7, color='green')
    axes[0, 1].axhline(y=results_df['r2'].mean(), color='red', linestyle='--',
                       label=f'Mean: {results_df["r2"].mean():.3f}')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('R² by Hour of Day')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actual vs Predicted demand by hour
    axes[1, 0].plot(results_df['hour'], results_df['mean_actual'], 
                    marker='o', label='Actual', linewidth=2)
    axes[1, 0].plot(results_df['hour'], results_df['mean_predicted'], 
                    marker='s', label='Predicted', linewidth=2)
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Average Pickup Count')
    axes[1, 0].set_title('Average Demand by Hour')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample distribution by hour
    axes[1, 1].bar(results_df['hour'], results_df['n_samples'], alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Sample Distribution by Hour')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_hour.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Hourly analysis saved")
    return results_df

def evaluate_by_day_of_week(y_true, y_pred, df, output_dir):
    """Evaluate performance by day of week"""
    results = []
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
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
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RMSE by day
    axes[0].bar(results_df['day_name'], results_df['rmse'], alpha=0.7, color='steelblue')
    axes[0].axhline(y=results_df['rmse'].mean(), color='red', linestyle='--',
                    label=f'Mean: {results_df["rmse"].mean():.2f}')
    axes[0].set_xlabel('Day of Week')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Day of Week')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R² by day
    axes[1].bar(results_df['day_name'], results_df['r2'], alpha=0.7, color='green')
    axes[1].axhline(y=results_df['r2'].mean(), color='red', linestyle='--',
                    label=f'Mean: {results_df["r2"].mean():.3f}')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('R²')
    axes[1].set_title('R² by Day of Week')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Actual vs Predicted
    x = np.arange(len(results_df))
    width = 0.35
    axes[2].bar(x - width/2, results_df['mean_actual'], width, label='Actual', alpha=0.7)
    axes[2].bar(x + width/2, results_df['mean_predicted'], width, label='Predicted', alpha=0.7)
    axes[2].set_xlabel('Day of Week')
    axes[2].set_ylabel('Average Pickup Count')
    axes[2].set_title('Average Demand by Day')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(results_df['day_name'], rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_day_of_week.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Day-of-week analysis saved")
    return results_df

def evaluate_by_zone_demand_level(y_true, y_pred, df, output_dir):
    """Evaluate by zone demand levels (low/medium/high)"""
    # Define demand levels
    demand_quantiles = df['zone_avg_demand'].quantile([0.33, 0.67])
    
    df_eval = df.copy()
    df_eval['demand_level'] = pd.cut(
        df_eval['zone_avg_demand'],
        bins=[-np.inf, demand_quantiles[0.33], demand_quantiles[0.67], np.inf],
        labels=['Low Demand', 'Medium Demand', 'High Demand']
    )
    
    results = []
    
    for level in ['Low Demand', 'Medium Demand', 'High Demand']:
        mask = df_eval['demand_level'] == level
        if mask.sum() > 0:
            y_l = y_true[mask]
            pred_l = y_pred[mask]
            
            results.append({
                'demand_level': level,
                'rmse': np.sqrt(mean_squared_error(y_l, pred_l)),
                'mae': mean_absolute_error(y_l, pred_l),
                'r2': r2_score(y_l, pred_l),
                'n_samples': mask.sum(),
                'mean_actual': y_l.mean(),
                'mean_predicted': pred_l.mean(),
                'std_actual': y_l.std(),
                'std_predicted': pred_l.std()
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_by_zone_demand.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    # RMSE
    axes[0].bar(results_df['demand_level'], results_df['rmse'], 
                alpha=0.7, color=colors)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Zone Demand Level')
    axes[0].grid(True, alpha=0.3)
    
    # R²
    axes[1].bar(results_df['demand_level'], results_df['r2'], 
                alpha=0.7, color=colors)
    axes[1].set_ylabel('R²')
    axes[1].set_title('R² by Zone Demand Level')
    axes[1].grid(True, alpha=0.3)
    
    # Actual vs Predicted
    x = np.arange(len(results_df))
    width = 0.35
    axes[2].bar(x - width/2, results_df['mean_actual'], width, 
                label='Actual', alpha=0.7)
    axes[2].bar(x + width/2, results_df['mean_predicted'], width, 
                label='Predicted', alpha=0.7)
    axes[2].set_ylabel('Average Pickup Count')
    axes[2].set_title('Average Demand by Zone Level')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(results_df['demand_level'])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_zone_demand.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Zone demand level analysis saved")
    return results_df

def evaluate_peak_vs_offpeak(y_true, y_pred, df, output_dir):
    """Compare peak hours vs off-peak"""
    results = []
    
    # Peak hours
    peak_mask = df['is_rush_hour'] == 1
    if peak_mask.sum() > 0:
        y_peak = y_true[peak_mask]
        pred_peak = y_pred[peak_mask]
        
        results.append({
            'period': 'Peak Hours',
            'rmse': np.sqrt(mean_squared_error(y_peak, pred_peak)),
            'mae': mean_absolute_error(y_peak, pred_peak),
            'r2': r2_score(y_peak, pred_peak),
            'n_samples': peak_mask.sum(),
            'mean_actual': y_peak.mean(),
            'mean_predicted': pred_peak.mean()
        })
    
    # Off-peak hours
    offpeak_mask = df['is_rush_hour'] == 0
    if offpeak_mask.sum() > 0:
        y_off = y_true[offpeak_mask]
        pred_off = y_pred[offpeak_mask]
        
        results.append({
            'period': 'Off-Peak Hours',
            'rmse': np.sqrt(mean_squared_error(y_off, pred_off)),
            'mae': mean_absolute_error(y_off, pred_off),
            'r2': r2_score(y_off, pred_off),
            'n_samples': offpeak_mask.sum(),
            'mean_actual': y_off.mean(),
            'mean_predicted': pred_off.mean()
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_peak_vs_offpeak.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RMSE comparison
    axes[0].bar(results_df['period'], results_df['rmse'], 
                alpha=0.7, color=['#e74c3c', '#3498db'])
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE: Peak vs Off-Peak')
    axes[0].grid(True, alpha=0.3)
    
    # R² comparison
    axes[1].bar(results_df['period'], results_df['r2'], 
                alpha=0.7, color=['#e74c3c', '#3498db'])
    axes[1].set_ylabel('R²')
    axes[1].set_title('R²: Peak vs Off-Peak')
    axes[1].grid(True, alpha=0.3)
    
    # Demand comparison
    x = np.arange(len(results_df))
    width = 0.35
    axes[2].bar(x - width/2, results_df['mean_actual'], width, 
                label='Actual', alpha=0.7)
    axes[2].bar(x + width/2, results_df['mean_predicted'], width, 
                label='Predicted', alpha=0.7)
    axes[2].set_ylabel('Average Pickup Count')
    axes[2].set_title('Average Demand: Peak vs Off-Peak')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(results_df['period'])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_peak_vs_offpeak.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Peak vs off-peak analysis saved")
    return results_df

def evaluate_by_demand_magnitude(y_true, y_pred, df, output_dir):
    """Evaluate by actual demand magnitude (binned)"""
    # Create demand bins
    bins = [0, 10, 25, 50, 100, 200, np.inf]
    labels = ['0-10', '10-25', '25-50', '50-100', '100-200', '200+']
    
    df_eval = df.copy()
    df_eval['demand_bin'] = pd.cut(y_true, bins=bins, labels=labels)
    
    results = []
    
    for bin_label in labels:
        mask = df_eval['demand_bin'] == bin_label
        if mask.sum() > 10:  # Only if we have enough samples
            y_bin = y_true[mask]
            pred_bin = y_pred[mask]
            
            results.append({
                'demand_range': bin_label,
                'rmse': np.sqrt(mean_squared_error(y_bin, pred_bin)),
                'mae': mean_absolute_error(y_bin, pred_bin),
                'mape': np.mean(np.abs((y_bin - pred_bin) / (y_bin + 1))) * 100,  # +1 to avoid div by 0
                'n_samples': mask.sum(),
                'mean_actual': y_bin.mean(),
                'mean_predicted': pred_bin.mean()
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'metrics_by_demand_magnitude.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE
    axes[0, 0].bar(results_df['demand_range'], results_df['rmse'], alpha=0.7)
    axes[0, 0].set_xlabel('Actual Demand Range')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE by Demand Magnitude')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAPE
    axes[0, 1].bar(results_df['demand_range'], results_df['mape'], 
                   alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Actual Demand Range')
    axes[0, 1].set_ylabel('MAPE (%)')
    axes[0, 1].set_title('Mean Absolute Percentage Error')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample distribution
    axes[1, 0].bar(results_df['demand_range'], results_df['n_samples'], 
                   alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Actual Demand Range')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Sample Distribution by Demand Level')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Actual vs Predicted
    x = np.arange(len(results_df))
    width = 0.35
    axes[1, 1].bar(x - width/2, results_df['mean_actual'], width, 
                   label='Actual', alpha=0.7)
    axes[1, 1].bar(x + width/2, results_df['mean_predicted'], width, 
                   label='Predicted', alpha=0.7)
    axes[1, 1].set_xlabel('Actual Demand Range')
    axes[1, 1].set_ylabel('Average Pickup Count')
    axes[1, 1].set_title('Actual vs Predicted by Demand Level')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(results_df['demand_range'], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_demand_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Demand magnitude analysis saved")
    return results_df

def evaluate_top_zones(y_true, y_pred, df, output_dir, n_zones=10):
    """Evaluate performance on top N busiest zones"""
    # Get top zones by total demand
    zone_totals = df.groupby('PULocationID')['pickup_count'].sum().sort_values(ascending=False)
    top_zones = zone_totals.head(n_zones).index
    
    results = []
    
    for zone_id in top_zones:
        mask = df['PULocationID'] == zone_id
        if mask.sum() > 0:
            y_zone = y_true[mask]
            pred_zone = y_pred[mask]
            
            results.append({
                'zone_id': int(zone_id),
                'total_pickups': int(zone_totals[zone_id]),
                'rmse': np.sqrt(mean_squared_error(y_zone, pred_zone)),
                'mae': mean_absolute_error(y_zone, pred_zone),
                'r2': r2_score(y_zone, pred_zone),
                'n_samples': mask.sum()
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / f'metrics_top_{n_zones}_zones.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # RMSE by zone
    axes[0].barh(results_df['zone_id'].astype(str), results_df['rmse'], alpha=0.7)
    axes[0].set_xlabel('RMSE')
    axes[0].set_ylabel('Zone ID')
    axes[0].set_title(f'RMSE for Top {n_zones} Busiest Zones')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # R² by zone
    axes[1].barh(results_df['zone_id'].astype(str), results_df['r2'], 
                 alpha=0.7, color='green')
    axes[1].set_xlabel('R²')
    axes[1].set_ylabel('Zone ID')
    axes[1].set_title(f'R² for Top {n_zones} Busiest Zones')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'performance_top_{n_zones}_zones.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Top {n_zones} zones analysis saved")
    return results_df

def plot_error_heatmap(y_true, y_pred, df, output_dir):
    """Create heatmap of errors by hour and day of week"""
    df_eval = df.copy()
    df_eval['error'] = np.abs(y_true - y_pred)
    
    # Aggregate errors
    error_pivot = df_eval.groupby(['day_of_week', 'hour'])['error'].mean().reset_index()
    error_matrix = error_pivot.pivot(index='day_of_week', columns='hour', values='error')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 6))
    
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    sns.heatmap(error_matrix, annot=False, fmt='.1f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Mean Absolute Error'})
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    
    # Only set labels for days that exist in the data
    actual_days = error_matrix.index.tolist()
    actual_day_labels = [day_labels[int(d)] for d in actual_days]
    ax.set_yticklabels(actual_day_labels, rotation=0)
    
    ax.set_title('Mean Absolute Error Heatmap (Hour × Day of Week)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Error heatmap saved")

def generate_summary_report(output_dir):
    """Generate text summary of all analyses"""
    report_path = output_dir / 'granular_evaluation_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GRANULAR EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Overall performance
        if (output_dir / 'overall_performance.csv').exists():
            overall_df = pd.read_csv(output_dir / 'overall_performance.csv')
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 70 + "\n")
            f.write(overall_df.to_string(index=False))
            f.write("\n\n")
        
        # Load all metrics
        files = {
            'By Hour': 'metrics_by_hour.csv',
            'By Day of Week': 'metrics_by_day_of_week.csv',
            'By Zone Demand': 'metrics_by_zone_demand.csv',
            'Peak vs Off-Peak': 'metrics_peak_vs_offpeak.csv',
            'By Demand Magnitude': 'metrics_by_demand_magnitude.csv'
        }
        
        for title, filename in files.items():
            filepath = output_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                f.write(f"\n{title.upper()}:\n")
                f.write("-" * 70 + "\n")
                f.write(df.to_string(index=False))
                f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("KEY INSIGHTS:\n")
        f.write("="*70 + "\n\n")
        
        # Hour analysis
        if (output_dir / 'metrics_by_hour.csv').exists():
            hour_df = pd.read_csv(output_dir / 'metrics_by_hour.csv')
            worst_hours = hour_df.nlargest(3, 'rmse')[['hour', 'rmse']]
            best_hours = hour_df.nsmallest(3, 'rmse')[['hour', 'rmse']]
            
            f.write("WORST PERFORMING HOURS (Highest RMSE):\n")
            for _, row in worst_hours.iterrows():
                f.write(f"  Hour {int(row['hour']):2d}: RMSE = {row['rmse']:.2f}\n")
            
            f.write("\nBEST PERFORMING HOURS (Lowest RMSE):\n")
            for _, row in best_hours.iterrows():
                f.write(f"  Hour {int(row['hour']):2d}: RMSE = {row['rmse']:.2f}\n")
        
        # Peak vs off-peak
        if (output_dir / 'metrics_peak_vs_offpeak.csv').exists():
            peak_df = pd.read_csv(output_dir / 'metrics_peak_vs_offpeak.csv')
            f.write("\nPEAK VS OFF-PEAK PERFORMANCE:\n")
            for _, row in peak_df.iterrows():
                f.write(f"  {row['period']}: RMSE = {row['rmse']:.2f}, R² = {row['r2']:.4f}\n")
            
            # Calculate difference
            if len(peak_df) == 2:
                peak_rmse = peak_df[peak_df['period'] == 'Peak Hours']['rmse'].values[0]
                offpeak_rmse = peak_df[peak_df['period'] == 'Off-Peak Hours']['rmse'].values[0]
                diff = peak_rmse - offpeak_rmse
                pct_diff = (diff / offpeak_rmse) * 100
                f.write(f"\n  Peak hours are {pct_diff:.1f}% worse than off-peak\n")
        
        # Zone demand
        if (output_dir / 'metrics_by_zone_demand.csv').exists():
            zone_df = pd.read_csv(output_dir / 'metrics_by_zone_demand.csv')
            f.write("\nPERFORMANCE BY ZONE DEMAND LEVEL:\n")
            for _, row in zone_df.iterrows():
                f.write(f"  {row['demand_level']}: RMSE = {row['rmse']:.2f}, R² = {row['r2']:.4f}\n")
        
        # Day of week
        if (output_dir / 'metrics_by_day_of_week.csv').exists():
            dow_df = pd.read_csv(output_dir / 'metrics_by_day_of_week.csv')
            worst_day = dow_df.loc[dow_df['rmse'].idxmax()]
            best_day = dow_df.loc[dow_df['rmse'].idxmin()]
            f.write("\nDAY OF WEEK ANALYSIS:\n")
            f.write(f"  Worst day: {worst_day['day_name']} (RMSE = {worst_day['rmse']:.2f})\n")
            f.write(f"  Best day: {best_day['day_name']} (RMSE = {best_day['rmse']:.2f})\n")
        
        # Overall accuracy summary
        if (output_dir / 'overall_performance.csv').exists():
            overall_df = pd.read_csv(output_dir / 'overall_performance.csv')
            test_row = overall_df[overall_df['split'] == 'Test'].iloc[0]
            f.write("\nACCURACY SUMMARY (Test Set):\n")
            f.write(f"  {test_row['acc_within_10']:.1f}% of predictions within ±10 pickups\n")
            f.write(f"  {test_row['acc_within_20pct']:.1f}% of predictions within ±20% error\n")
            f.write(f"  Mean Absolute Percentage Error: {test_row['mape']:.2f}%\n")
    
    print(f"  ✓ Summary report saved to: {report_path}")

def granular_evaluate_model(model_path, output_dir=None):
    """
    Comprehensive granular evaluation
    """
    model_path = Path(model_path)
    
    if output_dir is None:
        output_dir = model_path.parent / 'granular_evaluation'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GRANULAR EVALUATION PIPELINE")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model and data...")
    model = load_any_model(model_path)
    
    # Load data
    train_df = pd.read_parquet('data/processed/train.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')
    
    print(f"  ✓ Train: {len(train_df):,} samples")
    print(f"  ✓ Test:  {len(test_df):,} samples")
    
    # STEP 0: Overall performance evaluation
    train_metrics, test_metrics, y_pred = evaluate_overall_performance(
        model, train_df, test_df, output_dir
    )
    
    # Get test features for granular analysis
    X_test, y_test = prepare_features(test_df)
    
    # Run all granular evaluations
    print(f"{'='*70}")
    print("GRANULAR ANALYSIS (Test Set Only)")
    print(f"{'='*70}\n")
    
    print("1. Evaluating by hour of day...")
    evaluate_by_hour(y_test, y_pred, test_df, output_dir)
    
    print("\n2. Evaluating by day of week...")
    evaluate_by_day_of_week(y_test, y_pred, test_df, output_dir)
    
    print("\n3. Evaluating by zone demand level...")
    evaluate_by_zone_demand_level(y_test, y_pred, test_df, output_dir)
    
    print("\n4. Evaluating peak vs off-peak...")
    evaluate_peak_vs_offpeak(y_test, y_pred, test_df, output_dir)
    
    print("\n5. Evaluating by demand magnitude...")
    evaluate_by_demand_magnitude(y_test, y_pred, test_df, output_dir)
    
    print("\n6. Evaluating top zones...")
    evaluate_top_zones(y_test, y_pred, test_df, output_dir, n_zones=15)
    
    print("\n7. Creating error heatmap...")
    plot_error_heatmap(y_test, y_pred, test_df, output_dir)
    
    print("\n8. Generating summary report...")
    generate_summary_report(output_dir)
    
    print(f"\n{'='*70}")
    print("GRANULAR EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey files generated:")
    print("  overall_performance.csv")
    print("  overall_performance_comparison.png")
    print("  metrics_by_hour.csv & plot")
    print("  metrics_by_day_of_week.csv & plot")
    print("  metrics_by_zone_demand.csv & plot")
    print("  metrics_peak_vs_offpeak.csv & plot")
    print("  metrics_by_demand_magnitude.csv & plot")
    print("  metrics_top_15_zones.csv & plot")
    print("  error_heatmap.png")
    print("  granular_evaluation_summary.txt")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("QUICK SUMMARY")
    print(f"{'='*70}")
    print(f"\nOverall Performance:")
    print(f"  Train RMSE: {train_metrics['rmse']:.2f} | Test RMSE: {test_metrics['rmse']:.2f}")
    print(f"  Train R²:   {train_metrics['r2']:.4f} | Test R²:   {test_metrics['r2']:.4f}")
    print(f"  Test Accuracy (±10 pickups): {test_metrics['acc_within_10']:.2f}%")
    
    # Load and show peak vs off-peak
    if (output_dir / 'metrics_peak_vs_offpeak.csv').exists():
        peak_df = pd.read_csv(output_dir / 'metrics_peak_vs_offpeak.csv')
        print(f"\nPeak Hour Performance:")
        for _, row in peak_df.iterrows():
            print(f"  {row['period']}: RMSE = {row['rmse']:.2f}, R² = {row['r2']:.4f}")
    
    # Show worst performing hours
    if (output_dir / 'metrics_by_hour.csv').exists():
        hour_df = pd.read_csv(output_dir / 'metrics_by_hour.csv')
        worst_hours = hour_df.nlargest(3, 'rmse')
        print(f"\nWorst 3 Hours (Highest RMSE):")
        for _, row in worst_hours.iterrows():
            print(f"  Hour {int(row['hour']):2d}: RMSE = {row['rmse']:.2f}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'models/baseline/model.pkl'
    
    granular_evaluate_model(model_path)