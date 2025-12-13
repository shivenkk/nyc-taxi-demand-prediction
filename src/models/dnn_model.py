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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class DNNNet(nn.Module):
    def __init__(self, input_size, hidden_layers=[128, 64, 32], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h
        
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze()


class DNNWrapper:
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values
        
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.FloatTensor(X_scaled).to(device)).cpu().numpy()
        return preds


def load_data():
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    test = pd.read_parquet('data/processed/test.parquet')
    return train, val, test


def get_feature_cols(df):
    exclude = ['pickup_count', 'PULocationID', 'pickup_hour']
    return [c for c in df.columns if c not in exclude]


def main():
    print("DNN Model Training")
    
    HIDDEN = [128, 64, 32]
    DROPOUT = 0.1
    BATCH = 256
    EPOCHS = 100
    LR = 0.001
    PATIENCE = 15
    
    print("\nLoading data...")
    train, val, test = load_data()
    feature_cols = get_feature_cols(train)
    print(f"Features: {len(feature_cols)}")
    
    X_train, y_train = train[feature_cols].values, train['pickup_count'].values
    X_val, y_val = val[feature_cols].values, val['pickup_count'].values
    X_test, y_test = test[feature_cols].values, test['pickup_count'].values
    
    print("Scaling...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train)),
        batch_size=BATCH, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val)),
        batch_size=BATCH
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_s), torch.FloatTensor(y_test)),
        batch_size=BATCH
    )
    
    print(f"\nModel: {HIDDEN}, dropout={DROPOUT}")
    model = DNNNet(len(feature_cols), HIDDEN, DROPOUT).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print("\nTraining...")
    best_val_rmse = float('inf')
    best_state = None
    patience_ctr = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_preds.extend(model(X_b.to(device)).cpu().numpy())
                val_targets.extend(y_b.numpy())
        
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)
        scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict().copy()
            patience_ctr = 0
            marker = " *"
        else:
            patience_ctr += 1
            marker = ""
        
        if (epoch + 1) % 10 == 0 or marker:
            print(f"Epoch {epoch+1}: Val RMSE={val_rmse:.2f}, R2={val_r2:.4f}{marker}")
        
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Best Val RMSE: {best_val_rmse:.2f}")
    model.load_state_dict(best_state)
    
    print("\nTest evaluation:")
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            test_preds.extend(model(X_b.to(device)).cpu().numpy())
            test_targets.extend(y_b.numpy())
    
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)
    test_mae = mean_absolute_error(test_targets, test_preds)
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R2:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.2f}")
    
    output_dir = Path('models/dnn')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / 'dnn_weights.pt')
    wrapper = DNNWrapper(model, scaler, feature_cols)
    joblib.dump(wrapper, output_dir / 'model.pkl')
    
    metrics = [{'split': 'Test', 'rmse': test_rmse, 'r2': test_r2, 'mae': test_mae, 'n_samples': len(y_test)}]
    pd.DataFrame(metrics).to_csv(output_dir / 'evaluation_metrics.csv', index=False)
    
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
