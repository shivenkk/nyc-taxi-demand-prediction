import pandas as pd
from pathlib import Path

def load_raw_data(filepath: str = "data/raw/yellow_tripdata_2024-01.parquet") -> pd.DataFrame:
    """Load raw taxi trip data from parquet file."""
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['tpep_pickup_datetime'].min()} to {df['tpep_pickup_datetime'].max()}")
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
