import pandas as pd
from pathlib import Path

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw taxi data by removing invalid records like the ones with following cases:
    - Trips with invalid coordinates (PULocationID outside 1-263)
    - Negative fares
    - Trips > 100 miles or < 0.05 miles
    - Trips outside January 2024
    - Invalid passenger counts
    - Negative trip durations or trips > 3 hours    
    - Null values in critical columns
    """
    initial_count = len(df)
    removal_stats = {}

    # Keep only January 2024
    before = len(df)
    df = df[
        (df['tpep_pickup_datetime'] >= '2024-01-01') & 
        (df['tpep_pickup_datetime'] < '2024-02-01')
    ]
    removal_stats['Outside January 2024'] = before - len(df)

    # Check for nulls first
    critical_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 
                       'PULocationID', 'fare_amount', 'trip_distance']
    
    before = len(df)
    df = df.dropna(subset=critical_columns)
    removal_stats['Null values'] = before - len(df)

    # Filter to valid pickup locations (263 taxi zones)
    before = len(df)
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 263)]
    removal_stats['Invalid Zone IDs'] = before - len(df)

    # Remove zero or negative fares
    before = len(df)
    df = df[df['fare_amount'] > 0]
    removal_stats['Zero/Negative Fares'] = before - len(df)

    # Remove unreasonably long/short trips
    before = len(df)
    df = df[(df['trip_distance'] <= 100) & (df['trip_distance'] >= 0.05)]
    removal_stats['Invalid Distance (< 0.05 or > 100 miles)'] = before - len(df)

    # Remove negative durations or trips > 3 hours
    before = len(df)
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 10800)]  # 0 to 3 hours
    removal_stats['Invalid Trip Duration'] = before - len(df)
    df = df.drop('trip_duration', axis=1)

    # Remove invalid passenger counts
    before = len(df)
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    removal_stats['Invalid Passenger Count'] = before - len(df)

    stats_df = pd.DataFrame.from_dict(removal_stats, orient='index', columns=['Records Removed'])
    stats_df.index.name = 'Reason'
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats_path = output_dir / 'cleaning_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"Cleaning statistics saved to: {stats_path}")

    # Print detailed report
    print(f"\n{'='*60}")
    print(f"DATA CLEANING REPORT")
    print(f"{'='*60}")
    print(f"Initial records: {initial_count:,}")
    print(f"\nRecords removed by reason:")
    for reason, count in removal_stats.items():
        pct = (count / initial_count) * 100 if initial_count > 0 else 0
        print(f"  {reason}: {count:,} ({pct:.2f}%)")
    
    final_count = len(df)
    total_removed = initial_count - final_count
    print(f"\nFinal records: {final_count:,}")
    print(f"Total removed: {total_removed:,} ({(total_removed/initial_count)*100:.2f}%)")
    print(f"Data retention rate: {(final_count/initial_count)*100:.2f}%")
    print(f"{'='*60}\n")
    
    return df
    

if __name__ == "__main__":
    from load_data import load_raw_data
    
    df = load_raw_data()
    df_clean = clean_data(df)
    print(f"\nSample of cleaned data:")
    print(df_clean.head())
