import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import json

from lstm_utils import LSTMNet, LSTMWrapper, device

print(f"Using device: {device}")


def load_data():
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    test = pd.read_parquet('data/processed/test.parquet')
    return train, val, test


def get_feature_cols(df):
    exclude = ['pickup_count', 'PULocationID', 'pickup_hour']
    return [c for c in df.columns if c not in exclude]


def create_sequences(X, y, seq_length):
    sequences = []
    targets = []
    
    for i in range(seq_length, len(X)):
        seq = X[i-seq_length:i]
        sequences.append(seq)
        targets.append(y[i])
    
    return np.array(sequences), np.array(targets)


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
    
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(X_batch)
            loss = criterion(output, y_batch)
            
            total_loss += loss.item() * len(y_batch)
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    return avg_loss, rmse, r2, np.array(all_preds), np.array(all_targets)


def main():
    print("LSTM Model Training")
    
    # hyperparameters
    SEQ_LENGTH = 24
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001
    PATIENCE = 10
    
    # load data
    print("\nLoading data")
    train, val, test = load_data()
    
    train = train.sort_values(['PULocationID', 'pickup_hour']).reset_index(drop=True)
    val = val.sort_values(['PULocationID', 'pickup_hour']).reset_index(drop=True)
    test = test.sort_values(['PULocationID', 'pickup_hour']).reset_index(drop=True)
    
    feature_cols = get_feature_cols(train)
    print(f"Features: {len(feature_cols)}")
    
    X_train = train[feature_cols].values
    y_train = train['pickup_count'].values
    X_val = val[feature_cols].values
    y_val = val['pickup_count'].values
    X_test = test[feature_cols].values
    y_test = test['pickup_count'].values
    
    # scale features
    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # create sequences
    print(f"Creating sequences (length={SEQ_LENGTH})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQ_LENGTH)
    
    print(f"Train sequences: {X_train_seq.shape}")
    print(f"Val sequences: {X_val_seq.shape}")
    print(f"Test sequences: {X_test_seq.shape}")
    
    # dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # create model
    print(f"\nCreating LSTM model")
    model = LSTMNet(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # training
    print(f"\nTraining...")
    
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_rmse, val_r2, _, _ = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""
        
        if (epoch + 1) % 5 == 0 or marker:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.2f}, R2={val_r2:.4f}{marker}")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"Best Val RMSE: {best_val_rmse:.2f}")
    model.load_state_dict(best_model_state)
    
    # test eval
    print("\nTest set evaluation:")
    _, test_rmse, test_r2, test_preds, test_targets = evaluate(model, test_loader, criterion)
    test_mae = mean_absolute_error(test_targets, test_preds)
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R2:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.2f}")
    
    # save
    output_dir = Path('models/lstm')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / 'lstm_weights.pt')
    
    wrapper = LSTMWrapper(model, scaler, SEQ_LENGTH, feature_cols)
    joblib.dump(wrapper, output_dir / 'model.pkl')
    
    config = {
        'seq_length': SEQ_LENGTH,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'batch_size': BATCH_SIZE,
        'epochs_trained': epoch + 1,
        'best_val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2)
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # save metrics
    metrics = [
        {'split': 'Test', 'rmse': test_rmse, 'r2': test_r2, 'mae': test_mae, 'n_samples': len(y_test_seq)}
    ]
    pd.DataFrame(metrics).to_csv(output_dir / 'evaluation_metrics.csv', index=False)
    
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
