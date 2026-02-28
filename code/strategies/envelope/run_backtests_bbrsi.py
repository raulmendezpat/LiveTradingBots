#!/usr/bin/env python3
"""Run BB+RSI backtests for BTC and SOL in one command."""
from __future__ import annotations
import argparse
from pathlib import Path
from backtest_bbrsi import to_ms, fetch_ohlcv_ccxt, simulate, summary, parse_params_from_bot_file

def run_one(bot_file: str, start: str, end: str|None, exchange: str, initial: float, maker_fee: float, taker_fee: float, outdir: str):
    params=parse_params_from_bot_file(bot_file)
    symbol=params.get("symbol"); timeframe=params.get("timeframe","1h")
    df=fetch_ohlcv_ccxt(symbol, timeframe, to_ms(start), to_ms(end) if end else None, exchange)
    trades_df, equity_df = simulate(df, params, initial, maker_fee, taker_fee)

    out=Path(outdir); out.mkdir(parents=True, exist_ok=True)
    name=Path(bot_file).stem
    trades_path=out/f"{name}_trades.csv"; equity_path=out/f"{name}_equity.csv"
    trades_df.to_csv(trades_path, index=False); equity_df.to_csv(equity_path, index=False)

    print(f"\n=== {name} ===")
    for k,v in summary(trades_df,equity_df).items():
        print(f"{k}: {v}")
    print(f"Saved -> {trades_path}")
    print(f"Saved -> {equity_path}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--btc-bot", required=True)
    ap.add_argument("--sol-bot", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--outdir", default="results")
    args=ap.parse_args()

    run_one(args.btc_bot, args.start, args.end, args.exchange, args.initial, args.maker_fee, args.taker_fee, args.outdir)
    run_one(args.sol_bot, args.start, args.end, args.exchange, args.initial, args.maker_fee, args.taker_fee, args.outdir)

if __name__=="__main__":
    main()
