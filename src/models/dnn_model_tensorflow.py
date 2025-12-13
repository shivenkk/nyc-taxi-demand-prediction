import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

def scale_features(X_train, X_val, X_test=None):
    """
    Scale features using StandardScaler
    Neural networks require scaled inputs!
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    return X_train_scaled, X_val_scaled, scaler

def build_dnn_model(input_dim, 
                   hidden_layers=[128, 64, 32],
                   dropout_rate=0.2,
                   learning_rate=0.001,
                   activation='relu'):
    """
    Build a feed-forward neural network
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        activation: Activation function for hidden layers
    """
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation=activation, name=f'hidden_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer (regression)
    model.add(layers.Dense(1, activation='linear', name='output'))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_dnn_model(X_train, y_train, X_val, y_val,
                   hidden_layers=[128, 64, 32],
                   dropout_rate=0.2,
                   learning_rate=0.001,
                   batch_size=256,
                   epochs=100,
                   patience=15,
                   verbose=0):
    """
    Train a DNN model with early stopping
    """
    print(f"  Training DNN (layers={hidden_layers}, dropout={dropout_rate}, "
          f"lr={learning_rate})...", end=' ')
    
    start_time = time.time()
    
    # Build model
    model = build_dnn_model(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    actual_epochs = len(history.history['loss'])
    
    print(f"Val RMSE: {val_rmse:.2f}, Val R²: {val_r2:.4f} "
          f"(epochs: {actual_epochs}, time: {train_time:.1f}s)")
    
    return model, history, {
        'hidden_layers': str(hidden_layers),
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_trained': actual_epochs,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'train_time_seconds': train_time
    }

def hyperparameter_search(X_train, y_train, X_val, y_val):
    """
    Search for best DNN architecture and hyperparameters
    """
    print("HYPERPARAMETER SEARCH - Deep Neural Network")
    
    results = []
    models = []
    histories = []
    
    # Configuration strategy:
    # Test different architectures and regularization
    
    configs = [
        # Simple architectures
        {'hidden_layers': [64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 256},
        {'hidden_layers': [128, 64], 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 256},
        
        # Standard architectures
        {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 256},
        {'hidden_layers': [256, 128, 64], 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 256},
        
        # Deeper network
        {'hidden_layers': [128, 64, 32, 16], 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 256},
        
        # Different dropout rates
        {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.1, 'learning_rate': 0.001, 'batch_size': 256},
        {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.3, 'learning_rate': 0.001, 'batch_size': 256},
        
        # Different learning rates
        {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.0005, 'batch_size': 256},
        {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.002, 'batch_size': 256},
        
        # Different batch sizes
        {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 128},
        {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 512},
    ]
    
    print(f"Testing {len(configs)} configurations...\n")
    print("Note: Training with early stopping (patience=15 epochs)\n")
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] ", end='')
        model, history, metrics = train_dnn_model(
            X_train, y_train, X_val, y_val,
            **config,
            epochs=100,
            patience=15,
            verbose=0
        )
        results.append(metrics)
        models.append(model)
        histories.append(history)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best model
    best_idx = results_df['val_rmse'].idxmin()
    best_model = models[best_idx]
    best_history = histories[best_idx]
    
    print("HYPERPARAMETER SEARCH RESULTS")
    print(results_df.to_string(index=False))
    
    print(f"BEST MODEL:")
    print(f"  Architecture: {results_df.iloc[best_idx]['hidden_layers']}")
    print(f"  Dropout rate: {results_df.iloc[best_idx]['dropout_rate']}")
    print(f"  Learning rate: {results_df.iloc[best_idx]['learning_rate']}")
    print(f"  Batch size: {results_df.iloc[best_idx]['batch_size']}")
    print(f"  Epochs trained: {results_df.iloc[best_idx]['epochs_trained']}")
    print(f"  Validation RMSE: {results_df.iloc[best_idx]['val_rmse']:.2f}")
    print(f"  Validation R²: {results_df.iloc[best_idx]['val_r2']:.4f}")
    print(f"  Training time: {results_df.iloc[best_idx]['train_time_seconds']:.1f}s")
    
    return best_model, best_history, results_df

def plot_training_history(history, output_dir):
    """
    Plot training and validation loss over epochs
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Model Loss During Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error During Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved")

def save_results(best_model, results_df, scaler, scaler_y, best_history, feature_names, output_dir):
    """
    Save model, scaler, results, and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model (Keras format)
    model_path = output_dir / 'model.keras'
    best_model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save scaler (needed for predictions!)
    scaler_path = output_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to: {scaler_path}")

    # Save scaler (needed for predictions!)
    scaler_path = output_dir / 'scaler_y.pkl'
    joblib.dump(scaler_y, scaler_path)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save hyperparameter results
    hp_path = output_dir / 'hyperparameter_results.csv'
    results_df.to_csv(hp_path, index=False)
    print(f"✓ Hyperparameter results saved to: {hp_path}")
    
    # Save metadata
    best_config = results_df.iloc[results_df['val_rmse'].idxmin()].to_dict()
    
    metadata = {
        'model_name': 'Deep Neural Network (Feed-Forward)',
        'model_type': 'neural_network',
        'library': 'tensorflow/keras',
        'best_hyperparameters': {
            'hidden_layers': best_config['hidden_layers'],
            'dropout_rate': float(best_config['dropout_rate']),
            'learning_rate': float(best_config['learning_rate']),
            'batch_size': int(best_config['batch_size']),
            'epochs_trained': int(best_config['epochs_trained'])
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
        'n_features': len(feature_names),
        'feature_names': list(feature_names),
        'scaling': 'StandardScaler (mean=0, std=1)'
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {output_dir / 'metadata.json'}")
    
    # Plot training history
    plot_training_history(best_history, output_dir)

if __name__ == "__main__":
    print("DEEP NEURAL NETWORK (FEED-FORWARD) - TRAINING")
    
    # Check TensorFlow/GPU
    print("TensorFlow Configuration:")
    print(f"  Version: {tf.__version__}")
    print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"  GPU: {tf.config.list_physical_devices('GPU')}")
    print()
    
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
    
    # Scale features (CRITICAL for neural networks!)
    print("\nScaling features (StandardScaler)...")
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    print(f" Features scaled (mean=0, std=1)")
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()

    # Hyperparameter search
    best_model, best_history, results_df = hyperparameter_search(
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled
    )
    
    # Save everything
    print("SAVING RESULTS")
    
    output_dir = Path('models/dnn')
    save_results(best_model, results_df, scaler, y_scaler, best_history, X_train.columns, output_dir)
    
    print("TRAINING COMPLETE")
    print("\nNext steps:")
    print("  1. Run evaluation: python src/evaluation/evaluate_model.py models/dnn/model.keras \"Deep Neural Network\"")
    print("  2. Run granular evaluation: python src/evaluation/granular_evaluation.py models/dnn/model.keras")
    print("  3. Compare with baseline and Random Forest results")
