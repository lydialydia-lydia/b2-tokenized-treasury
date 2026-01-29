from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

from .paths import ensure_dirs, processed_dir


@dataclass
class BuildParams:
    ust_ticker: str = "BIL"       # UST proxy ETF
    btc_ticker: str = "BTC-USD"
    start: str = "2020-01-01"
    end: Optional[str] = None     # None -> today
    vol_window: int = 30
    ann_factor: int = 365         # crypto-style


def _require_yfinance() -> None:
    if yf is None:
        raise ImportError("yfinance not installed. pip install yfinance")


def _download_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    _require_yfinance()
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}")
    close = df["Close"].copy()
    close.name = ticker
    close.index = pd.to_datetime(close.index)
    return close


def build_market_dataset(params: BuildParams) -> dict:
    """
    Outputs (written to data/processed/):
      - market_prices.parquet: aligned daily close prices for UST proxy + BTC
      - market_features.parquet: simple engineered features (returns/logret/rolling vol)
      - data_dictionary.json: reproducibility metadata

    Assumptions:
      - BTC trades daily; UST proxy ETF trades business days.
      - We align onto a daily calendar and forward-fill ETF price on non-trading days.
      - Rolling vol uses ann_factor=365 by default (crypto convention). Use 252 for TradFi style.
    """
    ensure_dirs()
    out_dir = processed_dir()

    ust = _download_close(params.ust_ticker, params.start, params.end)
    btc = _download_close(params.btc_ticker, params.start, params.end)

    # Align to daily calendar covering both assets
    idx = pd.date_range(start=min(ust.index.min(), btc.index.min()),
                        end=max(ust.index.max(), btc.index.max()),
                        freq="D")
    prices = pd.concat([ust, btc], axis=1).reindex(idx)

    # Forward-fill ETF on non-trading days (BIL missing on weekends/holidays)
    prices[params.ust_ticker] = prices[params.ust_ticker].ffill()

    # BTC should be present most days; if there are gaps, forward-fill small gaps
    prices[params.btc_ticker] = prices[params.btc_ticker].ffill()

    # Basic features
    ret = prices.pct_change()
    logret = np.log(prices).diff()

    def roll_vol(x: pd.Series) -> pd.Series:
        return x.rolling(params.vol_window).std() * np.sqrt(params.ann_factor)

    vol = ret.apply(roll_vol)

    features = pd.DataFrame(index=prices.index)
    features[f"{params.ust_ticker}_ret"] = ret[params.ust_ticker]
    features[f"{params.btc_ticker}_ret"] = ret[params.btc_ticker]
    features[f"{params.ust_ticker}_logret"] = logret[params.ust_ticker]
    features[f"{params.btc_ticker}_logret"] = logret[params.btc_ticker]
    features[f"{params.ust_ticker}_vol"] = vol[params.ust_ticker]
    features[f"{params.btc_ticker}_vol"] = vol[params.btc_ticker]

    prices_path = os.path.join(out_dir, "market_prices.parquet")
    feats_path = os.path.join(out_dir, "market_features.parquet")
    dict_path = os.path.join(out_dir, "data_dictionary.json")

    prices.to_parquet(prices_path)
    features.to_parquet(feats_path)

    data_dict = {
        "tickers": {"ust_proxy": params.ust_ticker, "btc": params.btc_ticker},
        "date_range": {"start": str(prices.index.min().date()), "end": str(prices.index.max().date())},
        "params": {
            "start": params.start,
            "end": params.end,
            "vol_window": params.vol_window,
            "ann_factor": params.ann_factor,
        },
        "notes": [
            "UST proxy ETF is forward-filled over non-trading days to align with BTC daily frequency.",
            "Rolling volatility annualization uses sqrt(365) by default; use 252 for TradFi convention.",
        ],
    }
    with open(dict_path, "w") as f:
        json.dump(data_dict, f, indent=2)

    return {"prices_path": prices_path, "feats_path": feats_path, "dict_path": dict_path}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ust", default="BIL")
    p.add_argument("--btc", default="BTC-USD")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
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
    out = build_market_dataset(params)
    print("Saved:")
    for k, v in out.items():
        print(" -", k, v)


if __name__ == "__main__":
    main()
