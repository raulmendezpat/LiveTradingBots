#!/usr/bin/env python3
"""
Stress test: inject an extreme shock month into a portfolio monthly return series.

Reads:
- results/portfolio_monthly_*.csv (must have columns: month, return_pct, equity_end)

Modes:
1) --mode append: add a synthetic extra month at the end with return = shock_pct
2) --mode worst: place the shock month at the point that maximizes drawdown (worst-case timing)
3) --mode random: run many random placements of ONE shock month and report distribution

Outputs:
- prints summary for baseline and stressed series
- optionally saves stressed monthly series to CSV

Example:
python3 stress_test_shock.py \
  --monthly-csv results/portfolio_monthly_v3_50_50_btc2_5_sol3.csv \
  --initial 1000 --shock -15 --mode worst

Note: This is a *month-level* stress test. It does not model intramonth gaps.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

def equity_from_monthly(returns_pct: np.ndarray, initial: float) -> np.ndarray:
    eq = np.empty(len(returns_pct)+1, dtype=float)
    eq[0]=initial
    eq[1:] = initial * np.cumprod(1.0 + returns_pct/100.0)
    return eq

def max_drawdown(equity: np.ndarray) -> Tuple[float,float]:
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    mdd = float(dd.max()) if len(dd) else 0.0
    peak_at = float(peak[dd.argmax()]) if len(dd) else float(peak[-1])
    mdd_pct = (mdd/peak_at*100.0) if peak_at>0 else 0.0
    return mdd, mdd_pct

def summarize(returns_pct: np.ndarray, initial: float) -> dict:
    eq = equity_from_monthly(returns_pct, initial)
    end = float(eq[-1])
    ret = (end/initial - 1.0)*100.0
    mdd, mdd_pct = max_drawdown(eq)
    return {
        "end_equity": end,
        "return_pct": ret,
        "max_dd": mdd,
        "max_dd_pct": mdd_pct,
    }

def inject_worst(returns_pct: np.ndarray, shock_pct: float, initial: float) -> Tuple[np.ndarray,int,dict]:
    # Try inserting at every position (between months) and pick max drawdown
    best_idx = 0
    worst_mdd_pct = -1.0
    best_series = None
    best_sum = None
    for idx in range(len(returns_pct)+1):
        cand = np.insert(returns_pct, idx, shock_pct)
        s = summarize(cand, initial)
        if s["max_dd_pct"] > worst_mdd_pct:
            worst_mdd_pct = s["max_dd_pct"]
            best_idx = idx
            best_series = cand
            best_sum = s
    return best_series, best_idx, best_sum

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly-csv", required=True)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--shock", type=float, default=-15.0, help="shock month return in percent (e.g., -15)")
    ap.add_argument("--mode", choices=["append","worst","random"], default="worst")
    ap.add_argument("--sims", type=int, default=5000, help="for random mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None, help="optional output CSV for the stressed monthly series")
    args = ap.parse_args()

    mdf = pd.read_csv(args.monthly_csv)
    if "return_pct" not in mdf.columns:
        raise SystemExit("monthly CSV must contain 'return_pct'")
    base_r = mdf["return_pct"].dropna().to_numpy(dtype=float)

    base_sum = summarize(base_r, args.initial)
    print("\n=== Baseline (no shock) ===")
    for k,v in base_sum.items():
        print(f"{k}: {v}")

    if args.mode == "append":
        stressed = np.append(base_r, args.shock)
        ssum = summarize(stressed, args.initial)
        print(f"\n=== Stressed (append one shock month {args.shock:.2f}%) ===")
        for k,v in ssum.items():
            print(f"{k}: {v}")
        insert_info = {"insert_index": len(base_r), "note": "appended at end"}
    elif args.mode == "worst":
        stressed, idx, ssum = inject_worst(base_r, args.shock, args.initial)
        print(f"\n=== Stressed (worst-case timing, one shock month {args.shock:.2f}%) ===")
        print(f"insert_index: {idx} (0 means before first month, {len(base_r)} means after last month)")
        for k,v in ssum.items():
            print(f"{k}: {v}")
        insert_info = {"insert_index": idx, "note": "worst-case placement"}
    else:
        rng = np.random.default_rng(args.seed)
        n = len(base_r)
        out = np.empty((args.sims, 4), dtype=float)
        for i in range(args.sims):
            idx = int(rng.integers(0, n+1))
            stressed = np.insert(base_r, idx, args.shock)
            s = summarize(stressed, args.initial)
            out[i] = (s["end_equity"], s["return_pct"], s["max_dd"], s["max_dd_pct"])
        df = pd.DataFrame(out, columns=["end_equity","return_pct","max_dd","max_dd_pct"])
        print(f"\n=== Random placement of ONE shock month {args.shock:.2f}% (sims={args.sims}) ===")
        print(f"Prob(max_dd_pct >= 10%): {(df['max_dd_pct']>=10).mean()*100:.2f}%")
        for col in ["return_pct","max_dd_pct","end_equity"]:
            print(f"{col}: mean={df[col].mean():.2f}, p5={df[col].quantile(0.05):.2f}, p50={df[col].quantile(0.50):.2f}, p95={df[col].quantile(0.95):.2f}")
        # no single series to save
        return

    if args.save:
        # create a month label for the inserted shock
        months = mdf["month"].astype(str).tolist() if "month" in mdf.columns else [f"m{i+1}" for i in range(len(base_r))]
        shock_label = "SHOCK"
        months2 = months.copy()
        months2.insert(insert_info["insert_index"], shock_label)
        out_df = pd.DataFrame({"month": months2, "return_pct": stressed})
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.save, index=False)
        print(f"\nSaved stressed monthly series -> {args.save}")

if __name__ == "__main__":
    main()
