#!/usr/bin/env python3
"""
Monte Carlo (bootstrap) risk simulation for a portfolio backtest.

Two bootstrap modes:
1) Month-level bootstrap (recommended): sample monthly returns with replacement.
   This preserves the empirical distribution of month outcomes and roughly keeps
   regime clustering at the month granularity.
2) Trade-level bootstrap: sample trade PnLs with replacement.

Outputs:
- Probability(max_drawdown_pct >= threshold)
- Distribution stats for return, max DD, and ending equity
- Percentiles table
- Optionally saves a CSV of simulation results

Usage examples:

# Month-level (from portfolio_monthly_*.csv produced by backtest_portfolio_btc_sol.py)
python3 monte_carlo_portfolio.py \
  --monthly-csv results/portfolio_monthly_v3_50_50_btc2_5_sol3.csv \
  --initial 1000 --sims 5000 --dd-threshold 10

# Trade-level (from portfolio_trades_*.csv)
python3 monte_carlo_portfolio.py \
  --trades-csv results/portfolio_trades_v3_50_50_btc2_5_sol3.csv \
  --initial 1000 --sims 5000 --dd-threshold 10
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

@dataclass
class SimResult:
    end_equity: float
    return_pct: float
    max_dd: float
    max_dd_pct: float

def max_drawdown(equity: np.ndarray) -> Tuple[float, float]:
    """Return (max_dd_abs, max_dd_pct) using running peak."""
    if len(equity) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(dd.max())
    peak_at_max = float(peak[dd.argmax()]) if len(dd) else float(peak[-1])
    max_dd_pct = (max_dd / peak_at_max * 100.0) if peak_at_max > 0 else 0.0
    return max_dd, max_dd_pct

def bootstrap_monthly(monthly_returns: np.ndarray, initial: float, sims: int, seed: Optional[int]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(monthly_returns)
    out = np.empty((sims, 4), dtype=float)
    for i in range(sims):
        sampled = rng.choice(monthly_returns, size=n, replace=True)
        equity = np.empty(n + 1, dtype=float)
        equity[0] = initial
        # returns are percent
        equity[1:] = initial * np.cumprod(1.0 + sampled / 100.0)
        end_eq = float(equity[-1])
        ret = (end_eq / initial - 1.0) * 100.0
        mdd, mdd_pct = max_drawdown(equity)
        out[i] = (end_eq, ret, mdd, mdd_pct)
    return pd.DataFrame(out, columns=["end_equity", "return_pct", "max_dd", "max_dd_pct"])

def bootstrap_trades(trade_pnls: np.ndarray, initial: float, sims: int, seed: Optional[int]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(trade_pnls)
    out = np.empty((sims, 4), dtype=float)
    for i in range(sims):
        sampled = rng.choice(trade_pnls, size=n, replace=True)
        equity = np.empty(n + 1, dtype=float)
        equity[0] = initial
        equity[1:] = initial + np.cumsum(sampled)
        end_eq = float(equity[-1])
        ret = (end_eq / initial - 1.0) * 100.0
        mdd, mdd_pct = max_drawdown(equity)
        out[i] = (end_eq, ret, mdd, mdd_pct)
    return pd.DataFrame(out, columns=["end_equity", "return_pct", "max_dd", "max_dd_pct"])

def print_summary(df: pd.DataFrame, dd_threshold: float):
    prob = (df["max_dd_pct"] >= dd_threshold).mean() * 100.0
    print(f"\nProbability(max_drawdown_pct >= {dd_threshold:.2f}%) = {prob:.2f}%")

    def stats(col: str):
        s = df[col]
        return float(s.mean()), float(s.std(ddof=0)), float(s.min()), float(s.max())

    for col, label in [("return_pct","Return %"), ("max_dd_pct","Max DD %"), ("end_equity","End equity")]:
        m, sd, mn, mx = stats(col)
        print(f"{label}: mean={m:.2f}, std={sd:.2f}, min={mn:.2f}, max={mx:.2f}")

    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    q = df[["return_pct","max_dd_pct","end_equity"]].quantile([p/100 for p in pct])
    q.index = [f"p{p}" for p in pct]
    print("\nPercentiles:")
    print(q.to_string(float_format=lambda x: f"{x:.2f}"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly-csv", default=None, help="portfolio_monthly_*.csv (preferred)")
    ap.add_argument("--trades-csv", default=None, help="portfolio_trades_*.csv")
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--dd-threshold", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None, help="Optional path to save simulation results CSV")
    args = ap.parse_args()

    if (args.monthly_csv is None) == (args.trades_csv is None):
        raise SystemExit("Provide exactly one of --monthly-csv or --trades-csv")

    if args.monthly_csv:
        mdf = pd.read_csv(args.monthly_csv)
        if "return_pct" not in mdf.columns:
            raise SystemExit("monthly CSV must contain 'return_pct'")
        monthly_returns = mdf["return_pct"].dropna().to_numpy(dtype=float)
        if len(monthly_returns) < 6:
            raise SystemExit("Not enough monthly rows to bootstrap.")
        sims_df = bootstrap_monthly(monthly_returns, args.initial, args.sims, args.seed)
        mode = "MONTH"
    else:
        tdf = pd.read_csv(args.trades_csv)
        if "pnl" not in tdf.columns:
            raise SystemExit("trades CSV must contain 'pnl'")
        trade_pnls = tdf["pnl"].dropna().to_numpy(dtype=float)
        if len(trade_pnls) < 50:
            print("Warning: <50 trades, trade-level bootstrap may be unstable.")
        sims_df = bootstrap_trades(trade_pnls, args.initial, args.sims, args.seed)
        mode = "TRADE"

    print(f"Monte Carlo bootstrap mode: {mode}")
    print(f"Simulations: {args.sims}, Initial: {args.initial:.2f}, Seed: {args.seed}")
    print_summary(sims_df, args.dd_threshold)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        sims_df.to_csv(out, index=False)
        print(f"\nSaved simulations -> {out}")

if __name__ == "__main__":
    main()
