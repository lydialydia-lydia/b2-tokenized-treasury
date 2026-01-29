from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd

from .paths import processed_dir


@dataclass
class SimParams:
    initial_aum: float = 100_000_000
    initial_cash_ratio: float = 0.02

    # Redemption process
    mode: Literal["constant", "spike"] = "constant"
    daily_rate: float = 0.002
    spike_day: int = 120
    spike_rate: float = 0.08

    # Gate policy
    min_cash_ratio_gate: float = 0.005     # gate if cash/AUM falls below this
    max_daily_redemption_gate: float = 0.05 # gate if redemption demand exceeds this fraction of AUM

    # Liquidation waterfall
    sell_btc_first: bool = True
    haircut_ust: float = 0.0005  # 5 bps effective cost
    haircut_btc: float = 0.0020  # 20 bps effective cost

    # Slippage proxy (optional extra cost)
    slippage_bps_ust: float = 0.0
    slippage_bps_btc: float = 0.0


def load_inputs() -> tuple[pd.DataFrame, Dict[str, Any]]:
    proc = processed_dir()
    prices_path = os.path.join(proc, "market_prices.parquet")
    cfg_path = os.path.join(proc, "portfolio_config.json")

    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Missing {prices_path}. Provide processed data (commit to GitHub) or run 01.")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing {cfg_path}. Provide portfolio_config.json (commit) or run 02.")

    prices = pd.read_parquet(prices_path)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return prices, cfg


def simulate_issuer_balance_sheet(prices: pd.DataFrame, cfg: Dict[str, Any], params: SimParams) -> pd.DataFrame:
    """
    Returns a dataframe with:
      nav, cash_ratio, redemption_demand, redeemed_paid, redeemed_blocked,
      sold_ust_gross, sold_btc_gross, slippage_cost, gate

    Assumptions:
      - Portfolio holds UST proxy + BTC at target weights (no daily rebalancing here).
      - Redemptions are a fraction of current AUM. Pay with:
          1) cash
          2) sell assets with haircut/slippage
      - Gate triggers if cash_ratio < min_cash_ratio_gate OR redemption demand too large.
    """
    ust = cfg["assets"]["ust"]
    btc = cfg["assets"]["btc"]
    w = cfg["target_weights"]
    w_ust = float(w[ust])
    w_btc = float(w[btc])

    px = prices[[ust, btc]].dropna().copy()
    px.index = pd.to_datetime(px.index)

    # Portfolio mark-to-market index (constant weights)
    rets = px.pct_change().fillna(0.0)
    port_ret = w_ust * rets[ust] + w_btc * rets[btc]
    nav_index = (1.0 + port_ret).cumprod()

    # State variables
    aum0 = params.initial_aum
    cash = aum0 * params.initial_cash_ratio

    # Split non-cash holdings by weights
    inv0 = aum0 - cash
    ust_value = inv0 * w_ust
    btc_value = inv0 * w_btc

    rows = []

    for t, nav_mul in nav_index.items():
        # mark-to-market holdings by nav multiplier
        aum_pre = (cash + ust_value + btc_value) * (nav_mul / (rows[-1]["nav_mul"] if rows else 1.0))
        # scale non-cash by same multiplier (approx; keeps dynamics simple & stable)
        if rows:
            scale = nav_mul / rows[-1]["nav_mul"]
            ust_value *= scale
            btc_value *= scale

        # redemption demand
        if params.mode == "constant":
            redeem_rate = params.daily_rate
        else:
            redeem_rate = params.spike_rate if (len(rows) == params.spike_day) else params.daily_rate

        redemption_demand = redeem_rate * aum_pre

        # gate decision (based on pre-trade state)
        cash_ratio_pre = cash / max(aum_pre, 1e-12)
        gate = (cash_ratio_pre < params.min_cash_ratio_gate) or (redeem_rate > params.max_daily_redemption_gate)

        redeemed_paid = 0.0
        redeemed_blocked = 0.0
        sold_ust_gross = 0.0
        sold_btc_gross = 0.0
        slippage_cost = 0.0

        if gate:
            redeemed_blocked = redemption_demand
        else:
            # Pay from cash first
            pay = min(cash, redemption_demand)
            cash -= pay
            redeemed_paid += pay
            remaining = redemption_demand - pay

            # Then sell assets (waterfall)
            def sell(amount_needed: float, asset: str) -> float:
                nonlocal cash, ust_value, btc_value, sold_ust_gross, sold_btc_gross, slippage_cost
                if amount_needed <= 0:
                    return 0.0

                if asset == "ust":
                    avail = ust_value
                    gross = min(avail, amount_needed / (1.0 - params.haircut_ust))
                    ust_value -= gross
                    proceeds = gross * (1.0 - params.haircut_ust)
                    cash += proceeds
                    sold_ust_gross += gross
                    slippage_cost += gross * params.slippage_bps_ust / 1e4
                    return proceeds
                else:
                    avail = btc_value
                    gross = min(avail, amount_needed / (1.0 - params.haircut_btc))
                    btc_value -= gross
                    proceeds = gross * (1.0 - params.haircut_btc)
                    cash += proceeds
                    sold_btc_gross += gross
                    slippage_cost += gross * params.slippage_bps_btc / 1e4
                    return proceeds

            order = ["btc", "ust"] if params.sell_btc_first else ["ust", "btc"]

            for asset in order:
                if remaining <= 0:
                    break
                proceeds = sell(remaining, "btc" if asset == "btc" else "ust")
                # now cash increased; pay immediately
                pay2 = min(cash, remaining)
                cash -= pay2
                redeemed_paid += pay2
                remaining -= pay2

            if remaining > 0:
                # cannot meet full redemption -> treat as blocked (liquidity shortfall)
                redeemed_blocked += remaining

        aum_post = cash + ust_value + btc_value
        cash_ratio_post = cash / max(aum_post, 1e-12)

        rows.append({
            "date": t,
            "nav_mul": float(nav_mul),
            "nav": float(aum_post / aum0),
            "cash_ratio": float(cash_ratio_post),
            "redemption_demand": float(redemption_demand),
            "redeemed_paid": float(redeemed_paid),
            "redeemed_blocked": float(redeemed_blocked),
            "sold_ust_gross": float(sold_ust_gross),
            "sold_btc_gross": float(sold_btc_gross),
            "slippage_cost": float(slippage_cost),
            "gate": bool(gate),
        })

    out = pd.DataFrame(rows).set_index("date")
    return out


def main():
    prices, cfg = load_inputs()

    # default scenarios
    base = simulate_issuer_balance_sheet(prices, cfg, SimParams(mode="constant"))
    spike = simulate_issuer_balance_sheet(prices, cfg, SimParams(mode="spike"))

    proc = processed_dir()
    base_path = os.path.join(proc, "sim_baseline.parquet")
    spike_path = os.path.join(proc, "sim_spike.parquet")
    base.to_parquet(base_path)
    spike.to_parquet(spike_path)

    print("Saved:")
    print(" -", base_path)
    print(" -", spike_path)


if __name__ == "__main__":
    main()
