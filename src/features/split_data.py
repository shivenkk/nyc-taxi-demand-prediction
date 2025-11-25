import pandas as pd
from pathlib import Path

def load_features(filepath: str = "data/processed/features.parquet") -> pd.DataFrame:
    """Load feature-engineered data."""
    return pd.read_parquet(filepath)

def temporal_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Split data temporally to prevent data leakage with the following arguments:
        df: DataFrame with pickup_hour column
        train_ratio: proportion for training (default 70%)
        val_ratio: proportion for validation (default 15%)
        test_ratio is 15% remaining
    
    Returns:
        train_df, val_df, test_df
    """
    df = df.sort_values('pickup_hour')
    
    # here we get unique hours and find split points
    unique_hours = df['pickup_hour'].unique()
    n_hours = len(unique_hours)
    
    train_end_idx = int(n_hours * train_ratio)
    val_end_idx = int(n_hours * (train_ratio + val_ratio))
    
    train_end_time = unique_hours[train_end_idx]
    val_end_time = unique_hours[val_end_idx]
    
    # Split
    train_df = df[df['pickup_hour'] < train_end_time].copy()
    val_df = df[(df['pickup_hour'] >= train_end_time) & (df['pickup_hour'] < val_end_time)].copy()
    test_df = df[df['pickup_hour'] >= val_end_time].copy()
    
    print(f"Temporal split (70/15/15):")
    print(f"  Train: {len(train_df):,} samples | {df['pickup_hour'].min()} to {train_end_time}")
    print(f"  Val:   {len(val_df):,} samples | {train_end_time} to {val_end_time}")
    print(f"  Test:  {len(test_df):,} samples | {val_end_time} to {df['pickup_hour'].max()}")
    
    return train_df, val_df, test_df

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save train/val/test splits to parquet."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    print(f"\nSaved to data/processed/: train.parquet, val.parquet, test.parquet")

if __name__ == "__main__":
    
    # Load
    df = load_features()
    print(f"Loaded {len(df):,} samples\n")
    
    # Split
    train_df, val_df, test_df = temporal_split(df)
    
    # Save
    save_splits(train_df, val_df, test_df)
