from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

from .paths import processed_dir, ensure_dirs


@dataclass
class PortfolioParams:
    ust_col: str = "BIL"
    btc_col: str = "BTC-USD"
    w_ust: float = 0.95
    w_btc: float = 0.05
    rebalance_band: float = 0.01      # not used in constant-weight baseline, kept for later
    ann_factor: int = 365


def build_portfolio_config(params: PortfolioParams) -> str:
    """
    Writes:
      - data/processed/portfolio_config.json

    Assumptions:
      - Baseline portfolio uses constant weights (Treasury core + BTC sleeve).
      - Config file is used by balance-sheet simulator and redemption stress notebooks.
    """
    ensure_dirs()
    proc = processed_dir()
    prices_path = os.path.join(proc, "market_prices.parquet")
    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Missing {prices_path}. Run 01 (build_data.py) first or commit processed files.")

    prices = pd.read_parquet(prices_path)
    cols = list(prices.columns)

    if params.ust_col not in cols or params.btc_col not in cols:
        raise ValueError(f"Columns not found. Have={cols}, need ust={params.ust_col}, btc={params.btc_col}")

    cfg = {
        "assets": {"ust": params.ust_col, "btc": params.btc_col},
        "target_weights": {params.ust_col: params.w_ust, params.btc_col: params.w_btc},
        "rebalance_band": params.rebalance_band,
        "ann_factor": params.ann_factor,
        "mode": "constant",
        "notes": [
            "Baseline constant-weight allocation; later notebooks may implement rebalancing.",
        ],
    }

    out_path = os.path.join(proc, "portfolio_config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ust_col", default="BIL")
    p.add_argument("--btc_col", default="BTC-USD")
    p.add_argument("--w_ust", type=float, default=0.95)
    p.add_argument("--w_btc", type=float, default=0.05)
    p.add_argument("--rebalance_band", type=float, default=0.01)
    p.add_argument("--ann_factor", type=int, default=365)
    args = p.parse_args()

    out = build_portfolio_config(PortfolioParams(
        ust_col=args.ust_col,
        btc_col=args.btc_col,
        w_ust=args.w_ust,
        w_btc=args.w_btc,
        rebalance_band=args.rebalance_band,
        ann_factor=args.ann_factor,
    ))
    print("Saved:", out)


if __name__ == "__main__":
    main()
