import numpy as np
import torch
import torch.nn as nn
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMNet(nn.Module):
    """Simple LSTM for regression"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out.squeeze()


class LSTMWrapper:
    """Wrapper to make LSTM work with sklearn-style predict()"""
    def __init__(self, model, scaler, seq_length, feature_cols):
        self.model = model
        self.scaler = scaler
        self.seq_length = seq_length
        self.feature_cols = feature_cols
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values
        
        X_scaled = self.scaler.transform(X)
        
        # create sequences same way as training
        sequences = []
        valid_indices = []
        
        for i in range(len(X_scaled)):
            if i < self.seq_length - 1:
                # not enough history, pad with zeros
                pad_len = self.seq_length - 1 - i
                seq = np.vstack([
                    np.zeros((pad_len, X_scaled.shape[1])),
                    X_scaled[:i+1]
                ])
            else:
                seq = X_scaled[i - self.seq_length + 1:i + 1]
            
            sequences.append(seq)
            valid_indices.append(i)
        
        sequences = np.array(sequences)
        
        # predict in batches
        self.model.eval()
        predictions = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(device)
                preds = self.model(batch_tensor).cpu().numpy()
                predictions.extend(preds)
        
        return np.array(predictions)
