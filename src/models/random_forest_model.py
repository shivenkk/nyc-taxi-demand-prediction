import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import json
import time

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

def train_random_forest(X_train, y_train, X_val, y_val, 
                       n_estimators=100, max_depth=None, 
                       min_samples_split=2, min_samples_leaf=1,
                       max_features='sqrt', random_state=42):
    """
    Train a Random Forest model with given hyperparameters
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None = unlimited)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at leaf node
        max_features: Number of features to consider for best split
        random_state: Random seed for reproducibility
    """
    print(f"  Training RF (n_est={n_estimators}, max_depth={max_depth}, "
          f"min_split={min_samples_split}, max_feat={max_features})...", end=' ')
    
    start_time = time.time()
    
    # Create model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    # Train
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Val RMSE: {val_rmse:.2f}, Val R²: {val_r2:.4f} ({train_time:.1f}s)")
    
    return model, {
        'n_estimators': n_estimators,
        'max_depth': max_depth if max_depth else 'None',
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'train_time_seconds': train_time
    }

def hyperparameter_search_manual(X_train, y_train, X_val, y_val):
    """
    Manual hyperparameter search with focused configurations
    Faster than RandomizedSearchCV for structured search
    """
    print("HYPERPARAMETER SEARCH - Random Forest")
    
    results = []
    models = []
    
    # Configuration strategy:
    # Start simple, gradually increase complexity
    # Focus on configurations likely to work well for time series
    
    configs = [
        # Quick baseline
        {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 10, 'max_features': 'sqrt'},
        
        # Standard configurations
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 10, 'max_features': 'sqrt'},
        
        # More trees
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 2, 'max_features': 'sqrt'},
        
        # Different feature sampling
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 5, 'max_features': 'log2'},
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 5, 'max_features': 0.5},
        
        # Deep trees (may overfit, but let's try)
        {'n_estimators': 100, 'max_depth': 30, 'min_samples_split': 2, 'max_features': 'sqrt'},
        
        # Conservative (prevent overfitting)
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 20, 'max_features': 'sqrt'},
    ]
    
    print(f"Testing {len(configs)} configurations...\n")
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] ", end='')
        model, metrics = train_random_forest(X_train, y_train, X_val, y_val, **config)
        results.append(metrics)
        models.append(model)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best model
    best_idx = results_df['val_rmse'].idxmin()
    best_model = models[best_idx]
    
    print("HYPERPARAMETER SEARCH RESULTS")
    print(results_df.to_string(index=False))
    
    print(f"BEST MODEL:")
    print(f"  n_estimators: {results_df.iloc[best_idx]['n_estimators']}")
    print(f"  max_depth: {results_df.iloc[best_idx]['max_depth']}")
    print(f"  min_samples_split: {results_df.iloc[best_idx]['min_samples_split']}")
    print(f"  max_features: {results_df.iloc[best_idx]['max_features']}")
    print(f"  Validation RMSE: {results_df.iloc[best_idx]['val_rmse']:.2f}")
    print(f"  Validation R²: {results_df.iloc[best_idx]['val_r2']:.4f}")
    print(f"  Training time: {results_df.iloc[best_idx]['train_time_seconds']:.1f}s")
    
    return best_model, results_df

def analyze_feature_importance(model, feature_names, output_dir, top_n=20):
    """
    Analyze and visualize feature importance
    """
    import matplotlib.pyplot as plt
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices],
        'importance_pct': (importances[indices] / importances.sum()) * 100
    })
    
    # Save to CSV
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Print top features
    print(f"\nTop {top_n} Most Important Features:")
    for i in range(min(top_n, len(importance_df))):
        row = importance_df.iloc[i]
        print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f} ({row['importance_pct']:.2f}%)")
    
    # Plot feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top N features
    top_features = importance_df.head(top_n)
    axes[0].barh(range(len(top_features)), top_features['importance'])
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance')
    axes[0].set_title(f'Top {top_n} Most Important Features')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Cumulative importance
    axes[1].plot(range(1, len(importance_df) + 1), 
                 importance_df['importance_pct'].cumsum(), 
                 marker='o', markersize=3, linewidth=2)
    axes[1].axhline(y=80, color='r', linestyle='--', label='80% threshold')
    axes[1].axhline(y=90, color='orange', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('Cumulative Importance (%)')
    axes[1].set_title('Cumulative Feature Importance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Feature importance analysis saved")
    
    # Calculate how many features capture 80% and 90% of importance
    cum_importance = importance_df['importance_pct'].cumsum()
    n_80 = (cum_importance <= 80).sum() + 1
    n_90 = (cum_importance <= 90).sum() + 1
    
    print(f"\nFeature Selection Insights:")
    print(f"  {n_80} features capture 80% of importance")
    print(f"  {n_90} features capture 90% of importance")
    
    return importance_df

def save_results(best_model, results_df, feature_importance_df, X_train, output_dir):
    """
    Save model, results, and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'model.pkl'
    joblib.dump(best_model, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save hyperparameter results
    hp_path = output_dir / 'hyperparameter_results.csv'
    results_df.to_csv(hp_path, index=False)
    print(f"✓ Hyperparameter results saved to: {hp_path}")
    
    # Save metadata
    best_config = results_df.iloc[results_df['val_rmse'].idxmin()].to_dict()
    
    metadata = {
        'model_name': 'Random Forest Regressor',
        'model_type': 'ensemble_tree',
        'library': 'scikit-learn',
        'best_hyperparameters': {
            'n_estimators': int(best_config['n_estimators']),
            'max_depth': best_config['max_depth'],
            'min_samples_split': int(best_config['min_samples_split']),
            'min_samples_leaf': int(best_config['min_samples_leaf']),
            'max_features': best_config['max_features']
        },
        'performance': {
            'train_rmse': float(best_config['train_rmse']),
            'val_rmse': float(best_config['val_rmse']),
            'train_r2': float(best_config['train_r2']),
            'val_r2': float(best_config['val_r2']),
            'train_mae': float(best_config['train_mae']),
            'val_mae': float(best_config['val_mae'])
        },
        'training_time_seconds': float(best_config['train_time_seconds']),
        'n_features': len(feature_importance_df),
        'feature_names': list(X_train.columns),
        'top_5_features': feature_importance_df.head(5)['feature'].tolist()
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {output_dir / 'metadata.json'}")

if __name__ == "__main__":
    print("RANDOM FOREST REGRESSOR - TRAINING")
    
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
    print(f"\nFeature names:")
    for i, col in enumerate(X_train.columns, 1):
        print(f"    {i:2d}. {col}")
    
    # Hyperparameter search
    best_model, results_df = hyperparameter_search_manual(X_train, y_train, X_val, y_val)
    
    # Feature importance analysis
    print("FEATURE IMPORTANCE ANALYSIS")
    
    output_dir = Path('models/random_forest')
    importance_df = analyze_feature_importance(best_model, X_train.columns, output_dir, top_n=20)
    
    # Save everything
    print("SAVING RESULTS")
    
    save_results(best_model, results_df, importance_df, X_train, output_dir)
    
    print("TRAINING COMPLETE")
    print("\nNext steps:")
    print("  1. Run evaluation: python src/evaluation/evaluate_model.py models/random_forest/model.pkl \"Random Forest\"")
    print("  2. Run granular evaluation: python src/evaluation/granular_evaluation.py models/random_forest/model.pkl")
    print("  3. Compare with baseline results")
