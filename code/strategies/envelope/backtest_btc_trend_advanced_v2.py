#!/usr/bin/env python3
"""
Advanced backtester for BTC Trend Following 1H (EMA pullback + ADX filter + ATR TP/SL)
- Loads params from bot file (AST parse `params = {...}`)
- Uses ccxt to fetch OHLCV (Bitget public)
- Position sizing: balance * position_size_percentage * leverage
- Fees: maker on entry (limit), taker on exit (market)
- Intrabar TP/SL using candle high/low. If both hit in same candle -> conservative (SL first).
- Reports summary + month-by-month returns (%)
- Writes results/<bot>_trades.csv and results/<bot>_equity.csv and results/<bot>_monthly.csv
"""

from __future__ import annotations
import argparse, ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

# ---------- parsing ----------
def parse_params_from_bot_file(bot_file: str) -> Dict:
    src = Path(bot_file).read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src, filename=bot_file)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "params":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise ValueError("params is not a dict")
                    return value
    raise ValueError(f"Could not find `params = {{...}}` in {bot_file}")

def to_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_ohlcv_ccxt(symbol: str, timeframe: str, start_ms: int, end_ms: Optional[int], exchange_id: str = "bitget") -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Install with: pip install ccxt")
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    limit = 200
    rows = []
    since = start_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last == since:
            break
        since = last + 1
        if end_ms is not None and since >= end_ms:
            break
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if end_ms is not None:
        df = df[df["timestamp"] <= end_ms].reset_index(drop=True)
    return df

# ---------- indicators ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

# ---------- simulation ----------
@dataclass
class Trade:
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    reason: str
    fees: float

def month_key(ts_ms: int) -> str:
    return pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m")

def simulate(df: pd.DataFrame, params: Dict, initial_balance: float, maker_fee: float, taker_fee: float):
    df = df.copy()
    close = df["close"].astype(float)

    ema_fast_n = int(params.get("ema_fast", 20))
    ema_slow_n = int(params.get("ema_slow", 200))
    adx_period = int(params.get("adx_period", 14))
    adx_min = float(params.get("adx_min", 20))
    atr_period = int(params.get("atr_period", 14))

    stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
    tp_atr_mult = float(params.get("tp_atr_mult", 2.0))

    leverage = float(params.get("leverage", 2))
    pos_pct = float(params.get("position_size_percentage", 20)) / 100.0
    use_longs = bool(params.get("use_longs", True))
    use_shorts = bool(params.get("use_shorts", True))

    df["ema_fast"] = ema(close, ema_fast_n)
    df["ema_slow"] = ema(close, ema_slow_n)
    df["adx"] = adx(df, adx_period)
    df["atr"] = atr(df, atr_period)

    balance = initial_balance
    pos_side: Optional[str] = None
    entry_px = 0.0
    qty = 0.0
    entry_atr = 0.0

    trades: List[Trade] = []
    equity_rows = []

    for i in range(1, len(df)):
        prev = df.iloc[i-1]  # signal candle (closed)
        row = df.iloc[i]     # execution candle

        ts = int(row["timestamp"])
        h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

        # MTM equity snapshot
        mtm = 0.0
        if pos_side == "long":
            mtm = (c - entry_px) * qty
        elif pos_side == "short":
            mtm = (entry_px - c) * qty
        equity_rows.append({"timestamp": ts, "balance": balance, "equity": balance + mtm, "pos_side": pos_side or "", "qty": qty})

        # Require indicators on prev candle
        if any(prev[k] != prev[k] for k in ["ema_fast","ema_slow","adx","atr"]):
            continue

        # Manage open position (intrabar TP/SL)
        if pos_side is not None:
            tp = entry_px + tp_atr_mult * entry_atr if pos_side == "long" else entry_px - tp_atr_mult * entry_atr
            sl = entry_px - stop_atr_mult * entry_atr if pos_side == "long" else entry_px + stop_atr_mult * entry_atr

            exit_reason = None
            exit_price = None

            # Conservative: SL first if both touched same candle
            if l <= sl <= h:
                exit_reason = "sl"
                exit_price = sl
            elif l <= tp <= h:
                exit_reason = "tp"
                exit_price = tp

            if exit_reason and exit_price is not None:
                fee = exit_price * qty * taker_fee
                if pos_side == "long":
                    pnl = (exit_price - entry_px) * qty - fee
                    balance += (exit_price - entry_px) * qty
                else:
                    pnl = (entry_px - exit_price) * qty - fee
                    balance += (entry_px - exit_price) * qty
                balance -= fee

                trades.append(Trade(
                    side=pos_side,
                    entry_ts=int(prev["timestamp"]),
                    exit_ts=ts,
                    entry_price=entry_px,
                    exit_price=exit_price,
                    qty=qty,
                    pnl=pnl,
                    pnl_pct=pnl / (entry_px * qty) if entry_px > 0 else 0.0,
                    reason=exit_reason,
                    fees=fee,
                ))
                pos_side = None
                entry_px = 0.0
                qty = 0.0
                entry_atr = 0.0

            continue  # one action per candle

        # No position: entry conditions on prev candle; fill on current candle at EMA_fast (pullback limit)
        if float(prev["adx"]) < adx_min:
            continue

        ema_fast_v = float(prev["ema_fast"])
        ema_slow_v = float(prev["ema_slow"])
        close_v = float(prev["close"])
        atr_v = float(prev["atr"])
        if atr_v != atr_v or atr_v <= 0:
            continue

        notional = balance * pos_pct * leverage

        # Long pullback: uptrend and close below/at EMA_fast
        if use_longs and ema_fast_v > ema_slow_v and close_v <= ema_fast_v:
            entry = ema_fast_v
            if l <= entry <= h:
                qty = notional / entry
                balance -= notional * maker_fee
                pos_side = "long"
                entry_px = entry
                entry_atr = atr_v

        # Short pullback: downtrend and close above/at EMA_fast
        elif use_shorts and ema_fast_v < ema_slow_v and close_v >= ema_fast_v:
            entry = ema_fast_v
            if l <= entry <= h:
                qty = notional / entry
                balance -= notional * maker_fee
                pos_side = "short"
                entry_px = entry
                entry_atr = atr_v

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, equity_df

def summary(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
    if equity_df.empty:
        return {"error": "no equity"}
    start = float(equity_df["equity"].iloc[0])
    end = float(equity_df["equity"].iloc[-1])
    dd = equity_df["equity"].cummax() - equity_df["equity"]
    max_dd = float(dd.max()) if len(dd) else 0.0
    res = {
        "start_equity": start,
        "end_equity": end,
        "return_pct": (end / start - 1) * 100 if start else 0.0,
        "max_drawdown": max_dd,
        "max_drawdown_pct": (max_dd / float(equity_df["equity"].cummax().max())) * 100 if len(dd) else 0.0,
        "num_trades": int(len(trades_df)),
    }
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        res["win_rate_pct"] = (len(wins) / len(trades_df)) * 100
        res["avg_pnl"] = float(trades_df["pnl"].mean())
        res["profit_factor"] = float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else float("inf")
    return res

def monthly_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame(columns=["month", "return_pct", "equity_end"])

    tmp = equity_df.copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], unit="ms", utc=True)

    # Use year-month string directly (avoid timezone warnings)
    tmp["month"] = tmp["dt"].dt.strftime("%Y-%m")

    # Month-end equity
    month_end = tmp.groupby("month", as_index=False).tail(1)[["month", "equity"]].rename(columns={"equity": "equity_end"})
    month_end = month_end.sort_values("month").reset_index(drop=True)

    # Month-start equity is previous month-end (first month uses first equity sample)
    first_equity = float(tmp["equity"].iloc[0])
    month_end["equity_start"] = month_end["equity_end"].shift(1)
    month_end.loc[0, "equity_start"] = first_equity

    # Returns
    month_end["return_pct"] = (month_end["equity_end"] / month_end["equity_start"] - 1) * 100

    # Ensure no NaN rows
    month_end = month_end.dropna(subset=["month", "equity_end", "equity_start", "return_pct"]).reset_index(drop=True)

    return month_end[["month", "return_pct", "equity_end"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-file", required=True)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    params = parse_params_from_bot_file(args.bot_file)
    symbol = params.get("symbol")
    timeframe = params.get("timeframe", "1h")

    df = fetch_ohlcv_ccxt(symbol, timeframe, to_ms(args.start), to_ms(args.end) if args.end else None, args.exchange)

    trades_df, equity_df = simulate(df, params, args.initial, args.maker_fee, args.taker_fee)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.bot_file).stem

    trades_path = out / f"{name}_trades.csv"
    equity_path = out / f"{name}_equity.csv"
    monthly_path = out / f"{name}_monthly.csv"

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    monthly_df = monthly_returns(equity_df)
    monthly_df.to_csv(monthly_path, index=False)

    print(f"\n=== {name} ===")
    for k, v in summary(trades_df, equity_df).items():
        print(f"{k}: {v}")

    print("\n=== Monthly returns (%) ===")
    if monthly_df.empty:
        print("(no data)")
    else:
        for _, r in monthly_df.iterrows():
            print(f"{r['month']}: {r['return_pct']:.2f}% (equity_end={r['equity_end']:.2f})")

    print(f"\nSaved -> {trades_path}")
    print(f"Saved -> {equity_path}")
    print(f"Saved -> {monthly_path}")

if __name__ == "__main__":
    main()
