
import pandas as pd
import numpy as np
from pathlib import Path

def load_processed_data(filepath: str = "data/processed/hourly_demand.parquet") -> pd.DataFrame:
    """Load aggregated hourly demand data."""
    return pd.read_parquet(filepath)

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features from pickup_hour:
    - hour: 0-23
    - day_of_week: 0-6 (Monday=0)
    - is_weekend: binary
    - day_of_month: 1-31
    """
    df = df.copy()
    df['hour'] = df['pickup_hour'].dt.hour
    df['day_of_week'] = df['pickup_hour'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_month'] = df['pickup_hour'].dt.day
    
    print(f"Added temporal features: hour, day_of_week, is_weekend, day_of_month")
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for each zone:
    - lag_1h: pickup count 1hr ago
    - lag_2h: pickup count 2hrs ago
    - lag_24h: pickup count 24hrs ago
    """
    df = df.copy()
    df = df.sort_values(['PULocationID', 'pickup_hour'])
    
    for lag in [1, 2, 24]:
        df[f'lag_{lag}h'] = df.groupby('PULocationID')['pickup_count'].shift(lag)
    
    print(f"Added lag features: lag_1h, lag_2h, lag_24h")
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window features for each zone:
    - rolling_3h_mean: mean of last 3hrs
    - rolling_24h_mean: mean of last 24hrs
    """
    df = df.copy()
    df = df.sort_values(['PULocationID', 'pickup_hour'])
    
    # Rolling mean (shift by 1 to avoid data leakage while excluding current hour)
    df['rolling_3h_mean'] = df.groupby('PULocationID')['pickup_count'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['rolling_24h_mean'] = df.groupby('PULocationID')['pickup_count'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
    )
    
    print(f"Added rolling features: rolling_3h_mean, rolling_24h_mean")
    return df

def add_zone_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add zone-level aggregate features:
    - zone_avg_fare: historical avg fare for this zone
    - zone_avg_distance: historical avg trip distance for this zone
    - zone_avg_demand: historical avg hourly demand for this zone
    """
    df = df.copy()
    
    zone_stats = df.groupby('PULocationID').agg(
        zone_avg_fare=('avg_fare', 'mean'),
        zone_avg_distance=('avg_distance', 'mean'),
        zone_avg_demand=('pickup_count', 'mean')
    ).reset_index()
    
    df = df.merge(zone_stats, on='PULocationID', how='left')
    
    print(f"Added zone statistics: zone_avg_fare, zone_avg_distance, zone_avg_demand")
    return df

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_zone_statistics(df)
    
    # here we drop rows with NaN from lag features (basically only the first 24 hours per zone)
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    print(f"\nDropped {initial_count - final_count:,} rows with NaN (from lag features)")
    
    return df

def save_features(df: pd.DataFrame, filepath: str = "data/processed/features.parquet"):
    """Save feature-engineered data."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    print(f"Saved features to {filepath}")

if __name__ == "__main__":
    
    # Load
    df = load_processed_data()
    print(f"Loaded {len(df):,} samples")
    
    # Build features
    df = build_all_features(df)
    
    # Save
    save_features(df)
    
    # Summary
    print(f"\nFinal samples: {len(df):,}")
    print(f"\nFeatures ({len(df.columns)} columns):")
    print(df.columns.tolist())
    print(f"\nTarget stats (pickup_count):")
    print(df['pickup_count'].describe())
