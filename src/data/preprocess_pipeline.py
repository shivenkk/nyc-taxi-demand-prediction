"""
Full preprocessing pipeline to generate processed data: load -- clean -- aggregate -- save
"""
from load_data import load_raw_data
from clean_data import clean_data
from aggregate_data import aggregate_hourly, save_processed

def run_pipeline():
  
    # Load
    df = load_raw_data()
    
    # Clean
    df = clean_data(df)
    
    # Aggregate
    df_agg = aggregate_hourly(df)
    
    # Save
    save_processed(df_agg)
    
    print(f"Output: data/processed/hourly_demand.parquet")
    print(f"Samples: {len(df_agg):,}")
    
    return df_agg

if __name__ == "__main__":
    run_pipeline()
