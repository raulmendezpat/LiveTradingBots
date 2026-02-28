#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-bot", required=True, help="Path to BTC bot params .py (in this regime folder)")
    ap.add_argument("--sol-bot", required=True, help="Path to SOL bot params .py (in this regime folder)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--btc-weight", type=float, default=0.5)
    ap.add_argument("--sol-weight", type=float, default=0.5)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tag", default="regime_portfolio")
    ap.add_argument("--exchange", default=None)
    ap.add_argument("--maker-fee", type=float, default=None)
    ap.add_argument("--taker-fee", type=float, default=None)
    args = ap.parse_args()

    if abs((args.btc_weight + args.sol_weight) - 1.0) > 1e-6:
        raise SystemExit("btc-weight + sol-weight must equal 1.0")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Use sibling backtester so it works regardless of where you run from
    backtester = Path(__file__).with_name("backtest_regime.py")

    def call(bot: str):
        cmd = [sys.executable, str(backtester),
               "--bot-file", bot, "--start", args.start, "--initial", str(args.initial),
               "--outdir", str(outdir)]
        if args.end:
            cmd += ["--end", args.end]
        if args.exchange:
            cmd += ["--exchange", args.exchange]
        if args.maker_fee is not None:
            cmd += ["--maker-fee", str(args.maker_fee)]
        if args.taker_fee is not None:
            cmd += ["--taker-fee", str(args.taker_fee)]
        subprocess.check_call(cmd)

    call(args.btc_bot)
    call(args.sol_bot)

    btc_name = Path(args.btc_bot).stem
    sol_name = Path(args.sol_bot).stem

    btc_m = pd.read_csv(outdir / f"{btc_name}_monthly.csv")
    sol_m = pd.read_csv(outdir / f"{sol_name}_monthly.csv")

    merged = pd.merge(
        btc_m[["month", "return_pct"]].rename(columns={"return_pct": "r_btc"}),
        sol_m[["month", "return_pct"]].rename(columns={"return_pct": "r_sol"}),
        on="month",
        how="inner"
    ).sort_values("month").reset_index(drop=True)

    merged["return_pct"] = args.btc_weight * merged["r_btc"] + args.sol_weight * merged["r_sol"]

    eq = [args.initial]
    for r in merged["return_pct"].astype(float).tolist():
        eq.append(eq[-1] * (1.0 + r / 100.0))
    merged["equity_end"] = eq[1:]

    out_path = outdir / f"portfolio_monthly_{args.tag}.csv"
    merged[["month", "return_pct", "equity_end"]].to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
