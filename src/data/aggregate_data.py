import pandas as pd
from pathlib import Path

def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual trips into hourly pickup counts per zone and return DataFrame with columns:
    - PULocationID: pickup zone (1-263)
    - pickup_hour: datetime (hourly)
    - pickup_count: number of pickups (TARGET)
    - avg_fare: average fare for that zone-hour
    - avg_distance: average trip distance for that zone-hour
    """
    # Extract hour
    df = df.copy()
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')
    
    # Aggregate
    agg_df = df.groupby(['PULocationID', 'pickup_hour']).agg(
        pickup_count=('PULocationID', 'count'),
        avg_fare=('fare_amount', 'mean'),
        avg_distance=('trip_distance', 'mean')
    ).reset_index()
    
    print(f"Aggregated to {len(agg_df):,} zone-hour samples")
    print(f"Zones: {agg_df['PULocationID'].nunique()}")
    print(f"Hours: {agg_df['pickup_hour'].nunique()}")
    
    return agg_df

def save_processed(df: pd.DataFrame, filepath: str = "data/processed/hourly_demand.parquet"):
    """Save processed data to parquet."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    print(f"Saved to {filepath}")

if __name__ == "__main__":
    from load_data import load_raw_data
    from clean_data import clean_data
    
    df = load_raw_data()
    df = clean_data(df)
    df_agg = aggregate_hourly(df)
    save_processed(df_agg)
    
    print(f"\nTarget variable stats:")
    print(df_agg['pickup_count'].describe())

