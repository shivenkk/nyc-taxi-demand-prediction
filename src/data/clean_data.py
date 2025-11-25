import pandas as pd
from pathlib import Path

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw taxi data by removing invalid records like the ones with following cases:
    - Trips with invalid coordinates (PULocationID outside 1-263)
    - Negative fares
    - Trips > 100 miles
    - Trips outside January 2024
    """
    initial_count = len(df)
    
    # Filter to valid pickup locations (263 taxi zones)
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 263)]
    
    # Remove negative fares
    df = df[df['fare_amount'] >= 0]
    
    # Remove unreasonably long trips
    df = df[df['trip_distance'] <= 100]
    
    # Keep only January 2024
    df = df[
        (df['tpep_pickup_datetime'] >= '2024-01-01') & 
        (df['tpep_pickup_datetime'] < '2024-02-01')
    ]
    
    final_count = len(df)
    print(f"Cleaned: {initial_count:,} -> {final_count:,} records ({initial_count - final_count:,} removed)")
    
    return df

if __name__ == "__main__":
    from load_data import load_raw_data
    
    df = load_raw_data()
    df_clean = clean_data(df)
    print(f"\nSample of cleaned data:")
    print(df_clean.head())
