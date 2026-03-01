#!/usr/bin/env python3
"""
monte_carlo_portfolio.py (v2)

Enhancements:
- If no return_pct column exists, it computes trade return from:
      pnl / (abs(entry_price * qty)) * 100
- Compatible with your current trades CSV structure.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def equity_curve(returns_pct: np.ndarray, initial: float) -> np.ndarray:
    eq = np.empty(len(returns_pct) + 1)
    eq[0] = initial
    eq[1:] = initial * np.cumprod(1 + returns_pct / 100.0)
    return eq


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd.max() * 100)


def load_returns(trades_csv: str) -> np.ndarray:
    df = pd.read_csv(trades_csv)

    # Case 1: already has return column
    for c in ["return_pct", "pnl_pct", "trade_return_pct", "pct"]:
        if c in df.columns:
            return df[c].dropna().to_numpy(dtype=float)

    # Case 2: compute from pnl / (entry_price * qty)
    required = ["pnl", "entry_price", "qty"]
    if all(col in df.columns for col in required):
        df["notional"] = (df["entry_price"].astype(float) * df["qty"].astype(float)).abs()
        df["return_pct"] = df["pnl"].astype(float) / df["notional"] * 100.0
        return df["return_pct"].dropna().to_numpy(dtype=float)

    raise SystemExit(
        f"Cannot compute returns. Expected return column or columns {required}. Got {list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-csv", required=True)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--dd-threshold", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", type=str, default="mc_run")
    ap.add_argument("--save")
    args = ap.parse_args()

    returns = load_returns(args.trades_csv)
    n = len(returns)

    rng = np.random.default_rng(args.seed)
    results = np.zeros((args.sims, 3))

    for i in range(args.sims):
        shuffled = rng.permutation(returns)
        eq = equity_curve(shuffled, args.initial)
        final_eq = eq[-1]
        ret_pct = (final_eq / args.initial - 1) * 100
        mdd = max_drawdown(eq)
        results[i] = (final_eq, ret_pct, mdd)

    df = pd.DataFrame(results, columns=["end_equity", "return_pct", "max_dd_pct"])

    print(f"\n=== Monte Carlo: {args.name} ===")
    print(f"sims: {args.sims}")
    print(f"mean_return_pct: {df['return_pct'].mean():.2f}")
    print(f"p5_return_pct: {df['return_pct'].quantile(0.05):.2f}")
    print(f"p50_return_pct: {df['return_pct'].quantile(0.50):.2f}")
    print(f"p95_return_pct: {df['return_pct'].quantile(0.95):.2f}")
    print(f"mean_max_dd_pct: {df['max_dd_pct'].mean():.2f}")
    print(f"prob_dd_gt_{args.dd_threshold}%: {(df['max_dd_pct'] > args.dd_threshold).mean()*100:.2f}%")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save, index=False)
        print(f"Saved -> {args.save}")


if __name__ == "__main__":
    main()
