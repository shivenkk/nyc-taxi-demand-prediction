import pandas as pd
from pathlib import Path

def load_features(filepath: str = "data/processed/features.parquet") -> pd.DataFrame:
    """Load feature-engineered data."""
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} samples from {filepath}")
    return df

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
    df = df.sort_values('pickup_hour').reset_index(drop=True)
    
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
    
    # Verify no temporal overlap (data leakage check)
    assert train_df['pickup_hour'].max() < val_df['pickup_hour'].min(), "Train/Val temporal overlap detected!"
    assert val_df['pickup_hour'].max() < test_df['pickup_hour'].min(), "Val/Test temporal overlap detected!"
    print("Verified: No temporal overlap between splits (no data leakage)")

    return train_df, val_df, test_df

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save train/val/test splits to parquet."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    print(f"\nSaved to data/processed/: train.parquet, val.parquet, test.parquet")

    # Generate and save split summary CSV
    total_samples = len(train_df) + len(val_df) + len(test_df)
    
    summary_data = {
        'split': ['train', 'validation', 'test', 'total'],
        'samples': [len(train_df), len(val_df), len(test_df), total_samples],
        'percentage': [
            len(train_df)/total_samples*100,
            len(val_df)/total_samples*100,
            len(test_df)/total_samples*100,
            100.0
        ],
        'start_date': [
            str(train_df['pickup_hour'].min()),
            str(val_df['pickup_hour'].min()),
            str(test_df['pickup_hour'].min()),
            str(train_df['pickup_hour'].min())
        ],
        'end_date': [
            str(train_df['pickup_hour'].max()),
            str(val_df['pickup_hour'].max()),
            str(test_df['pickup_hour'].max()),
            str(test_df['pickup_hour'].max())
        ],
        'unique_zones': [
            train_df['PULocationID'].nunique(),
            val_df['PULocationID'].nunique(),
            test_df['PULocationID'].nunique(),
            263
        ],
        'target_mean': [
            round(train_df['pickup_count'].mean(), 2),
            round(val_df['pickup_count'].mean(), 2),
            round(test_df['pickup_count'].mean(), 2),
            round((train_df['pickup_count'].mean() + val_df['pickup_count'].mean() + test_df['pickup_count'].mean())/3, 2)
        ],
        'target_std': [
            round(train_df['pickup_count'].std(), 2),
            round(val_df['pickup_count'].std(), 2),
            round(test_df['pickup_count'].std(), 2),
            round((train_df['pickup_count'].std() + val_df['pickup_count'].std() + test_df['pickup_count'].std())/3, 2)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'split_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Split summary saved to: {summary_path}")
    
    # Save feature names for reference
    feature_cols = [col for col in train_df.columns if col not in ['pickup_hour', 'PULocationID', 'pickup_count']]
    feature_info = pd.DataFrame({
        'feature_name': feature_cols,
        'feature_type': [str(train_df[col].dtype) for col in feature_cols]
    })
    feature_path = output_dir / 'feature_list.csv'
    feature_info.to_csv(feature_path, index=False)
    print(f"Feature list saved to: {feature_path}")


if __name__ == "__main__":
    
    # Load
    df = load_features()
    print(f"Loaded {len(df):,} samples\n")
    
    # Split
    train_df, val_df, test_df = temporal_split(df)
    
    # Save
    save_splits(train_df, val_df, test_df)
