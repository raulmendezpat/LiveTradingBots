#!/usr/bin/env python3
"""
Monthly rebalance analysis for a 2-sleeve portfolio using existing portfolio equity output.

Input:
- portfolio_equity_*.csv produced by backtest_portfolio_btc_sol.py
  Must contain: timestamp, equity, equity_btc, equity_sol

What it does:
1) Computes month-end equity for BTC sleeve and SOL sleeve from the equity curve.
2) Computes monthly returns for each sleeve.
3) Builds two portfolio series:
   A) "as_is" (no rebalance): uses provided combined equity curve (equity column).
   B) "rebalance" (monthly): applies sleeve monthly returns and rebalances to target weights at each month-end.

Outputs (in results/ by default):
- rebalance_summary_<name>.txt (printed to console as well)
- rebalance_monthly_<name>.csv (month-by-month comparison)
- rebalance_equity_<name>.csv (monthly equity points for both series)

Note:
- This monthly rebalance simulation assumes sleeve returns scale cleanly with sleeve capital
  (reasonable here because both bots use % sizing and fees scale with notional).
"""

from __future__ import annotations
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import numpy as np

def summarize_equity(equity: pd.Series) -> Dict:
    if equity.empty:
        return {"error": "no equity"}
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    peak = equity.cummax()
    dd = peak - equity
    max_dd = float(dd.max()) if len(dd) else 0.0
    peak_at_max = float(peak.iloc[dd.idxmax()]) if len(dd) else float(peak.iloc[-1])
    max_dd_pct = (max_dd / peak_at_max * 100.0) if peak_at_max > 0 else 0.0
    return {
        "start_equity": start,
        "end_equity": end,
        "return_pct": (end / start - 1) * 100.0 if start else 0.0,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
    }

def month_end(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df[["timestamp", col]].copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], unit="ms", utc=True)
    tmp["month"] = tmp["dt"].dt.strftime("%Y-%m")
    me = tmp.groupby("month", as_index=False).tail(1)[["month", col]].rename(columns={col: f"{col}_end"})
    me = me.sort_values("month").reset_index(drop=True)
    return me

def monthly_returns_from_end(me: pd.DataFrame, end_col: str) -> pd.Series:
    end = me[end_col].astype(float)
    start = end.shift(1)
    start.iloc[0] = float(end.iloc[0])  # first month: define 0% return baseline
    r = (end / start - 1.0).fillna(0.0)
    return r

def build_rebalanced(months: pd.Series, r_btc: pd.Series, r_sol: pd.Series,
                     initial: float, w_btc: float, w_sol: float) -> pd.DataFrame:
    eq_total = []
    eq_btc = []
    eq_sol = []
    cur_btc = initial * w_btc
    cur_sol = initial * w_sol

    for i in range(len(months)):
        # apply monthly returns
        cur_btc = cur_btc * (1.0 + float(r_btc.iloc[i]))
        cur_sol = cur_sol * (1.0 + float(r_sol.iloc[i]))
        total = cur_btc + cur_sol

        # rebalance at month-end to target weights
        cur_btc = total * w_btc
        cur_sol = total * w_sol

        eq_total.append(total)
        eq_btc.append(cur_btc)
        eq_sol.append(cur_sol)

    return pd.DataFrame({
        "month": months,
        "equity_rebal_end": eq_total,
        "btc_rebal_end": eq_btc,
        "sol_rebal_end": eq_sol,
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-csv", required=True, help="results/portfolio_equity_<name>.csv")
    ap.add_argument("--btc-weight", type=float, default=0.5)
    ap.add_argument("--sol-weight", type=float, default=0.5)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--name", default="rebalance")
    args = ap.parse_args()

    if abs((args.btc_weight + args.sol_weight) - 1.0) > 1e-6:
        raise SystemExit("btc-weight + sol-weight must equal 1.0")

    df = pd.read_csv(args.equity_csv)
    for c in ["timestamp", "equity", "equity_btc", "equity_sol"]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    # month-end sleeve equities
    me_btc = month_end(df, "equity_btc")
    me_sol = month_end(df, "equity_sol")
    me_tot = month_end(df, "equity")

    # align months intersection
    months = pd.Series(sorted(set(me_btc["month"]).intersection(set(me_sol["month"])).intersection(set(me_tot["month"]))), name="month")
    me_btc = me_btc[me_btc["month"].isin(months)].reset_index(drop=True)
    me_sol = me_sol[me_sol["month"].isin(months)].reset_index(drop=True)
    me_tot = me_tot[me_tot["month"].isin(months)].reset_index(drop=True)

    r_btc = monthly_returns_from_end(me_btc, "equity_btc_end")
    r_sol = monthly_returns_from_end(me_sol, "equity_sol_end")

    # As-is (no rebalance): use combined month-end equity from file
    asis = pd.DataFrame({"month": me_tot["month"], "equity_asis_end": me_tot["equity_end"]})
    asis["return_asis_pct"] = (asis["equity_asis_end"].pct_change().fillna(0.0)) * 100.0

    # Rebalanced
    reb = build_rebalanced(months, r_btc, r_sol, args.initial, args.btc_weight, args.sol_weight)
    reb["return_rebal_pct"] = (reb["equity_rebal_end"].pct_change().fillna(0.0)) * 100.0

    out_monthly = asis.merge(reb, on="month", how="inner")

    # Build equity series (monthly points) for summary
    eq_asis = out_monthly["equity_asis_end"].astype(float)
    eq_reb = out_monthly["equity_rebal_end"].astype(float)

    s1 = summarize_equity(pd.concat([pd.Series([args.initial]), eq_asis], ignore_index=True))
    s2 = summarize_equity(pd.concat([pd.Series([args.initial]), eq_reb], ignore_index=True))

    print(f"\n=== Portfolio Monthly Rebalance Comparison: {args.name} ===")
    print(f"Target weights: BTC={args.btc_weight:.2f}, SOL={args.sol_weight:.2f}\n")

    print("As-is (no rebalance):")
    for k, v in s1.items():
        print(f"  {k}: {v}")

    print("\nMonthly rebalance:")
    for k, v in s2.items():
        print(f"  {k}: {v}")

    # save
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    monthly_path = outdir / f"rebalance_monthly_{args.name}.csv"
    equity_path = outdir / f"rebalance_equity_{args.name}.csv"
    out_monthly.to_csv(monthly_path, index=False)

    equity_points = pd.DataFrame({
        "month": out_monthly["month"],
        "equity_asis_end": eq_asis,
        "equity_rebal_end": eq_reb
    })
    equity_points.to_csv(equity_path, index=False)

    print(f"\nSaved -> {monthly_path}")
    print(f"Saved -> {equity_path}")

if __name__ == "__main__":
    main()
