import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for modeling.
    Excludes identifiers and target variable.
    """
    exclude_cols = ['pickup_count', 'PULocationID', 'pickup_hour']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['pickup_count'].copy()
    
    return X, y

def train_linear_model(X_train, y_train, X_val, y_val, 
                      model_type='ridge', alpha=1.0, degree=1):
    """
    Train linear regression model with optional polynomial features
    """
    # Choose model
    if model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha, max_iter=5000)
    else:
        model = LinearRegression()
    
    # Create pipeline
    if degree > 1:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
    # Train
    print(f"  Training {model_type} (degree={degree}, alpha={alpha})...", end=' ')
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Val RMSE: {val_rmse:.2f}, Val R²: {val_r2:.4f}")
    
    return pipeline, {
        'model_type': model_type,
        'alpha': alpha,
        'degree': degree,
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': val_rmse,
        'train_r2': r2_score(y_train, train_pred),
        'val_r2': val_r2,
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred)
    }

def hyperparameter_search(X_train, y_train, X_val, y_val):
    """
    Try different hyperparameters and return best model
    """
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SEARCH")
    print(f"{'='*70}\n")
    
    results = []
    models = []
    
    # Trying different configurations
    configs = [
        ('linear', 0.0, 1),      # Plain linear regression
        ('ridge', 0.1, 1),
        ('ridge', 1.0, 1),
        ('ridge', 10.0, 1),
        ('ridge', 1.0, 2),       # Polynomial degree 2
        ('lasso', 0.1, 1),
        ('lasso', 1.0, 1),
    ]
    
    for model_type, alpha, degree in configs:
        model, metrics = train_linear_model(X_train, y_train, X_val, y_val,
                                           model_type, alpha, degree)
        results.append(metrics)
        models.append(model)
    
    results_df = pd.DataFrame(results)
    
    # Find best model
    best_idx = results_df['val_rmse'].idxmin()
    best_model = models[best_idx]
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    print(results_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"BEST: {results_df.iloc[best_idx]['model_type']} "
          f"(alpha={results_df.iloc[best_idx]['alpha']}, degree={results_df.iloc[best_idx]['degree']})")
    print(f"  Val RMSE: {results_df.iloc[best_idx]['val_rmse']:.2f}")
    print(f"  Val R²: {results_df.iloc[best_idx]['val_r2']:.4f}")
    print(f"{'='*70}\n")
    
    return best_model, results_df

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("LINEAR REGRESSION BASELINE MODEL - TRAINING")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data splits...")
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val: {len(val_df):,} samples")
    
    # Prepare features
    print("\nPreparing features...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    
    print(f"  Features: {X_train.shape[1]}")
    
    # Hyperparameter search
    best_model, results_df = hyperparameter_search(X_train, y_train, X_val, y_val)
    
    # Save model
    output_dir = Path('models/baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'model.pkl'
    joblib.dump(best_model, model_path)
    print(f"Model saved to: {model_path}")
    
    hp_path = output_dir / 'hyperparameter_results.csv'
    results_df.to_csv(hp_path, index=False)
    print(f"Hyperparameter results saved to: {hp_path}")
    
    # Save metadata
    metadata = {
        'model_name': 'Linear Regression Baseline',
        'model_type': 'linear',
        'best_config': results_df.iloc[results_df['val_rmse'].idxmin()].to_dict(),
        'n_features': X_train.shape[1],
        'train_samples': len(X_train),
        'val_samples': len(X_val)
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {output_dir / 'metadata.json'}")
    
    print(f"\n{'='*70}")
    print("BASELINE MODEL TRAINING COMPLETE")
    print(f"{'='*70}")