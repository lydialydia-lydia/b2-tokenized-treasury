import numpy as np
import pandas as pd

def clean_and_align(prices: pd.DataFrame, freq="D") -> pd.DataFrame:
    prices = prices.copy()
    prices.columns = [c.replace("-USD", "").replace(" ", "_") for c in prices.columns]
    full_idx = pd.date_range(prices.index.min(), prices.index.max(), freq=freq)
    prices = prices.reindex(full_idx).ffill().dropna(how="all")
    return prices

def add_features(prices: pd.DataFrame, vol_window=30, ann_factor=365) -> pd.DataFrame:
    feats = pd.DataFrame(index=prices.index)
    for col in prices.columns:
        feats[f"{col}_ret"] = prices[col].pct_change()
        feats[f"{col}_logret"] = np.log(prices[col]).diff()
        feats[f"{col}_vol_{vol_window}d"] = feats[f"{col}_logret"].rolling(vol_window).std() * np.sqrt(ann_factor)
    return feats
