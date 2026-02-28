#!/usr/bin/env python3
"""
Run backtests for BOTH bots (BTC + SOL) in one command.

Example:
  python run_backtests.py \
    --btc-bot code/strategies/envelope/run_btc_1h_both.py \
    --sol-bot code/strategies/envelope/run_sol_1h_both.py \
    --start 2025-01-01 --end 2026-02-28 --initial 1000

This script calls backtest_envelope.simulate_envelope for each bot.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from backtest_envelope import (
    parse_params_from_bot_file,
    to_ms,
    fetch_ohlcv_ccxt,
    simulate_envelope,
    summary,
)

def run_one(bot_file: str, start: str, end: str | None, exchange: str, initial: float, maker_fee: float, taker_fee: float, outdir: str):
    params = parse_params_from_bot_file(bot_file)
    symbol = params.get("symbol")
    timeframe = params.get("timeframe", "1h")
    if not symbol:
        raise ValueError(f"Missing symbol in {bot_file}")

    start_ms = to_ms(start)
    end_ms = to_ms(end) if end else None
    df = fetch_ohlcv_ccxt(symbol, timeframe, start_ms, end_ms, exchange)

    trades_df, equity_df, _ = simulate_envelope(
        df=df, params=params, initial_balance=initial, maker_fee=maker_fee, taker_fee=taker_fee
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(bot_file).stem
    trades_path = out / f"{name}_trades.csv"
    equity_path = out / f"{name}_equity.csv"
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    print(f"\n=== {name} ===")
    for k, v in summary(trades_df, equity_df).items():
        print(f"{k}: {v}")
    print(f"Saved -> {trades_path}")
    print(f"Saved -> {equity_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-bot", required=True)
    ap.add_argument("--sol-bot", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    run_one(args.btc_bot, args.start, args.end, args.exchange, args.initial, args.maker_fee, args.taker_fee, args.outdir)
    run_one(args.sol_bot, args.start, args.end, args.exchange, args.initial, args.maker_fee, args.taker_fee, args.outdir)

if __name__ == "__main__":
    main()
