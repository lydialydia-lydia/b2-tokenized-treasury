"""
Build the processed market dataset for the B2 risk engine.

Outputs (written to data/processed/):
- market_prices.parquet: aligned daily prices for UST proxy + BTC
- market_features.parquet: baseline engineered features
- data_dictionary.json: reproducibility metadata

Assumptions:
- UST proxy is an ETF (trades business days) while BTC trades daily.
- We align to a daily calendar and forward-fill ETF prices on non-trading days.
- Rolling vol uses ann_factor=365 by default (crypto-style). Use 252 if you prefer TradFi.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Optional dependency: yfinance (installed in Colab cell when needed)
try:
    import yfinance as yf
except Exception:
    yf = None


@dataclass
class BuildParams:
    ust_ticker: str = "BIL"
    btc_ticker: str = "BTC-USD"
    start: str = "2020-01-01"
    end: Optional[str] = None
    freq: str = "D"
    vol_window: int = 30
    ann_factor: int = 365


def _require_yfinance():
    if yf is None:
        raise ImportError(
            "yfinance is not installed. In Colab run: pip install yfinance pyarrow"
        )


def fetch_adj_close(tickers: Sequence[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """
    Fetch auto-adjusted Close prices from Yahoo Finance.
    Returns a DataFrame with a DatetimeIndex.
    """
    _require_yfinance()
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px.index = pd.to_datetime(px.index)
    return px.sort_index()


def clean_and_align(prices: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Align all series to a daily calendar and forward-fill missing values.
    This is useful because BTC trades daily while ETFs do not.
    """
    prices = prices.copy()
    prices.columns = [c.replace("-USD", "").replace(" ", "_") for c in prices.columns]

    full_idx = pd.date_range(prices.index.min(), prices.index.max(), freq=freq)
    prices = prices.reindex(full_idx).ffill().dropna(how="all")
    prices.index.name = "date"
    return prices


def add_features(prices: pd.DataFrame, vol_window: int = 30, ann_factor: int = 365) -> pd.DataFrame:
    """
    Create baseline features used across later notebooks:
    - simple returns
    - log returns
    - rolling annualized volatility
    """
    feats = pd.DataFrame(index=prices.index)
    for col in prices.columns:
        feats[f"{col}_ret"] = prices[col].pct_change()
        feats[f"{col}_logret"] = np.log(prices[col]).diff()
        feats[f"{col}_vol_{vol_window}d"] = feats[f"{col}_logret"].rolling(vol_window).std() * np.sqrt(ann_factor)
    return feats


def build_and_save(out_dir: str, params: BuildParams) -> None:
    os.makedirs(out_dir, exist_ok=True)

    raw = fetch_adj_close([params.ust_ticker, params.btc_ticker], start=params.start, end=params.end)
    prices = clean_and_align(raw, freq=params.freq)

    feats = add_features(prices, vol_window=params.vol_window, ann_factor=params.ann_factor).dropna()
    prices = prices.loc[feats.index].copy()

    prices_path = os.path.join(out_dir, "market_prices.parquet")
    feats_path = os.path.join(out_dir, "market_features.parquet")
    dict_path = os.path.join(out_dir, "data_dictionary.json")

    prices.to_parquet(prices_path)
    feats.to_parquet(feats_path)

    data_dict = {
        "files": {
            "market_prices.parquet": {
                "index": "Daily date index (aligned with forward-fill)",
                "columns": list(prices.columns),
                "shape": list(prices.shape),
            },
            "market_features.parquet": {
                "index": "Daily date index (same as prices)",
                "columns": list(feats.columns),
                "shape": list(feats.shape),
            },
        },
        "tickers": {"ust_proxy": params.ust_ticker, "btc_proxy": params.btc_ticker},
        "date_range": {"start": str(prices.index.min().date()), "end": str(prices.index.max().date())},
        "params": {
            "freq": params.freq,
            "vol_window": params.vol_window,
            "ann_factor": params.ann_factor,
        },
        "notes": [
            "ETF/UST proxy is forward-filled over non-trading days to align with BTC daily frequency.",
            "Rolling volatility annualization uses sqrt(365) by default. Use 252 for TradFi convention.",
        ],
    }
    with open(dict_path, "w") as f:
        json.dump(data_dict, f, indent=2)

    print("Saved:")
    print(" -", prices_path, prices.shape)
    print(" -", feats_path, feats.shape)
    print(" -", dict_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ust", default="BIL", help="UST proxy ticker (ETF)")
    p.add_argument("--btc", default="BTC-USD", help="BTC proxy ticker")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="data/processed", help="output directory")
    p.add_argument("--vol_window", type=int, default=30)
    p.add_argument("--ann_factor", type=int, default=365)
    args = p.parse_args()

    params = BuildParams(
        ust_ticker=args.ust,
        btc_ticker=args.btc,
        start=args.start,
        end=args.end,
        vol_window=args.vol_window,
        ann_factor=args.ann_factor,
    )
    build_and_save(args.out, params)


if __name__ == "__main__":
    main()
