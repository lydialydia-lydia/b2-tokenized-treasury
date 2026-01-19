import pandas as pd

def load_processed_prices(proc_dir: str) -> pd.DataFrame:
    return pd.read_parquet(f"{proc_dir}/market_prices.parquet")

def load_processed_features(proc_dir: str) -> pd.DataFrame:
    return pd.read_parquet(f"{proc_dir}/market_features.parquet")
