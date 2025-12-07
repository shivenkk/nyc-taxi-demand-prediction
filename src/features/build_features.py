
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
    - is_rush_hour: binary (morning: 7-9am, evening: 5-7pm)
    """
    df = df.copy()
    df['hour'] = df['pickup_hour'].dt.hour
    df['day_of_week'] = df['pickup_hour'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_month'] = df['pickup_hour'].dt.day

    # Rush hour indicators (critical for taxi demand)
    df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype(int)
    df['is_evening_rush'] = df['hour'].isin([17, 18, 19]).astype(int)
    df['is_rush_hour'] = ((df['is_morning_rush'] == 1) | (df['is_evening_rush'] == 1)).astype(int)
       
    print(f"Added temporal features: hour, day_of_week, is_weekend, day_of_month")
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for each zone:
    - lag_1h: pickup count 1hr ago
    - lag_2h: pickup count 2hrs ago
    - lag_24h: pickup count 24hrs ago
    - lag_168h: pickup count 168hrs ago (same hour last week) 
   """
    df = df.copy()
    df = df.sort_values(['PULocationID', 'pickup_hour'])
    
    lags = [1, 2, 24]

    # Add weekly lag only if we have enough data (more than 7 days)
    date_range_days = (df['pickup_hour'].max() - df['pickup_hour'].min()).days
    if date_range_days >= 7:
        lags.append(168)  # 7 days * 24 hours

    for lag in lags:
        df[f'lag_{lag}h'] = df.groupby('PULocationID')['pickup_count'].shift(lag)
    
    print(f"Added lag features: {', '.join([f'lag_{l}h' for l in lags])}")
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window features for each zone:
    - rolling_3h_mean: mean of last 3hrs
    - rolling_24h_mean: mean of last 24hrs
    - rolling_3h_std: std dev of last 3hrs (captures volatility)
    - rolling_24h_std: std dev of last 24hrs
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
     
    # Rolling standard deviation (captures demand volatility)
    df['rolling_3h_std'] = df.groupby('PULocationID')['pickup_count'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
    )
    df['rolling_24h_std'] = df.groupby('PULocationID')['pickup_count'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1).std()
    )
       
    # Fill NaN std with 0 (happens when all values in window are the same)
    df['rolling_3h_std'] = df['rolling_3h_std'].fillna(0)
    df['rolling_24h_std'] = df['rolling_24h_std'].fillna(0)
    
    print(f"Added rolling features: rolling_3h_mean, rolling_24h_mean, rolling_3h_std, rolling_24h_std")
    return df

def add_zone_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add zone-level aggregate features:
    - zone_avg_fare: historical avg fare for this zone
    - zone_avg_distance: historical avg trip distance for this zone
    - zone_avg_demand: historical avg hourly demand for this zone
    - zone_demand_std: standard deviation of demand (high variability zones)
    """
    df = df.copy()
    
    zone_stats = df.groupby('PULocationID').agg(
        zone_avg_fare=('avg_fare', 'mean'),
        zone_avg_distance=('avg_distance', 'mean'),
        zone_avg_demand=('pickup_count', 'mean'),
        zone_demand_std=('pickup_count', 'std')
   ).reset_index()
    
    # Fill NaN std with 0 (zones with constant demand)
    zone_stats['zone_demand_std'] = zone_stats['zone_demand_std'].fillna(0)
    
    df = df.merge(zone_stats, on='PULocationID', how='left')
    
    print(f"Added zone statistics: zone_avg_fare, zone_avg_distance, zone_avg_demand")
    return df

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING PIPELINE")
    print(f"{'='*60}")
    print(f"Initial samples: {len(df):,}\n")
    
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_zone_statistics(df)
    
    # here we drop rows with NaN from lag features (basically only the first 24 hours per zone)
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    dropped = initial_count - final_count
    
    print(f"\nDropped {dropped:,} rows with NaN ({(dropped/initial_count)*100:.2f}%)")
    print(f"Final samples: {final_count:,}")
    print(f"{'='*60}\n")
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
