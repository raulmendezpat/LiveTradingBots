#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def equity_from_monthly(returns_pct: np.ndarray, initial: float) -> np.ndarray:
    eq = np.empty(len(returns_pct) + 1, dtype=float)
    eq[0] = initial
    eq[1:] = initial * np.cumprod(1.0 + returns_pct / 100.0)
    return eq


def max_drawdown(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    mdd = float(dd.max()) if len(dd) else 0.0
    peak_at = float(peak[dd.argmax()]) if len(dd) else float(peak[-1])
    mdd_pct = (mdd / peak_at * 100.0) if peak_at > 0 else 0.0
    return mdd, mdd_pct


def bootstrap_monthly(monthly_returns: np.ndarray, initial: float, sims: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(monthly_returns)
    out = np.empty((sims, 4), dtype=float)
    for i in range(sims):
        sampled = rng.choice(monthly_returns, size=n, replace=True)
        eq = equity_from_monthly(sampled, initial)
        end = float(eq[-1])
        ret = (end / initial - 1) * 100.0
        mdd, mdd_pct = max_drawdown(eq)
        out[i] = (end, ret, mdd, mdd_pct)
    return pd.DataFrame(out, columns=["end_equity", "return_pct", "max_dd", "max_dd_pct"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly-csv", required=True)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--dd-threshold", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    mdf = pd.read_csv(args.monthly_csv)
    r = mdf["return_pct"].dropna().to_numpy(dtype=float)

    df = bootstrap_monthly(r, args.initial, args.sims, args.seed)
    prob = (df["max_dd_pct"] >= args.dd_threshold).mean() * 100.0

    print(f"\nMonte Carlo (MONTH bootstrap)")
    print(f"Simulations: {args.sims}, Initial: {args.initial:.2f}")
    print(f"Probability(max_dd_pct >= {args.dd_threshold:.2f}%) = {prob:.2f}%")

    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    q = df[["return_pct", "max_dd_pct", "end_equity"]].quantile([p / 100 for p in pct])
    q.index = [f"p{p}" for p in pct]
    print("\nPercentiles:")
    print(q.to_string(float_format=lambda x: f"{x:.2f}"))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved simulations -> {out}")


if __name__ == "__main__":
    main()
