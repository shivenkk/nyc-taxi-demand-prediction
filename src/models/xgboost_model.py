import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path

def load_data():
    """Load train/val/test splits"""
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    test = pd.read_parquet('data/processed/test.parquet')
    return train, val, test

def prepare_features(df):
    """Separate features from target"""
    exclude = ['pickup_count', 'PULocationID', 'pickup_hour']
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y = df['pickup_count']
    return X, y

def train_and_evaluate(X_train, y_train, X_val, y_val, params):
    """Train a single XGBoost model and return validation metrics"""
    model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2 = r2_score(y_val, val_pred)
    
    return model, rmse, r2

def hyperparameter_search(X_train, y_train, X_val, y_val):
    """
    Try different hyperparameter combos and find best one. We save time by trying only sensible values.
    """
    results = []
    
    # different configs to try
    # format: (n_estimators, max_depth, learning_rate)
    configs = [
        (100, 3, 0.1),
        (100, 5, 0.1),
        (100, 7, 0.1),
        (200, 5, 0.1),
        (200, 5, 0.05),
        (300, 5, 0.05),
        (200, 7, 0.05),
    ]
    
    best_rmse = float('inf')
    best_model = None
    best_config = None
    
    for n_est, depth, lr in configs:
        params = {
            'n_estimators': n_est,
            'max_depth': depth,
            'learning_rate': lr,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        
        model, rmse, r2 = train_and_evaluate(X_train, y_train, X_val, y_val, params)
        
        print(f"n_est={n_est}, depth={depth}, lr={lr} --> Val RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        results.append({
            'n_estimators': n_est,
            'max_depth': depth,
            'learning_rate': lr,
            'val_rmse': rmse,
            'val_r2': r2
        })
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_config = params
    
    print(f"Best config: n_est={best_config['n_estimators']}, depth={best_config['max_depth']}, lr={best_config['learning_rate']}")
    print(f"Best Val RMSE: {best_rmse:.2f}")
    
    return best_model, pd.DataFrame(results), best_config

def main():
    print("XGBoost Model Training")
    
    # load data
    print("\nLoading data")
    train, val, test = load_data()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)
    print(f"Features: {X_train.shape[1]}")
    
    # hyperparameter search
    best_model, results_df, best_config = hyperparameter_search(X_train, y_train, X_val, y_val)
    
    # final evaluation on test set
    print("\nTest set evaluation:")
    test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R2:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.2f}")
    
    # feature importance (useful for report)
    print("\nTop 10 important features:")
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10).to_string(index=False))
    
    # save everything
    output_dir = Path('models/xgboost')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save model
    model_path = output_dir / 'model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # save hyperparam results
    results_df.to_csv(output_dir / 'hyperparameter_results.csv', index=False)
    
    # save feature importance
    importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # save config
    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
if __name__ == "__main__":
    main()
