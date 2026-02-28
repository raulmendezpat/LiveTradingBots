#!/usr/bin/env python3
"""Backtester: Bollinger + RSI mean reversion (single entry, re-entry after TP/SL)

- Loads params from a bot file (AST parse of `params = {...}`).
- Fetches OHLCV via ccxt OR reads CSV.
- Entry: if (ADX < adx_max) and close touches BB band and RSI extreme:
    * Long: close <= bb_low and rsi <= rsi_long_max
    * Short: close >= bb_up  and rsi >= rsi_short_min
  Then attempts to fill a limit at the band price on the NEXT candle (conservative).
- Exit:
    * TP at BB mid (basis) if touched
    * SL at entry +/- stop_atr_mult * ATR (ATR on signal candle)
  If TP and SL both touched same candle => conservative worst-case (SL first).
- Position sizing: notional = balance * pos_pct * leverage, qty = notional / entry_price
- Fees: maker on entry (limit), taker on exit (trigger-market). You can adjust.

Outputs:
- results/<bot>_trades.csv
- results/<bot>_equity.csv
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

def ensure_timestamp_ms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain 'timestamp'")
    if np.issubdtype(df["timestamp"].dtype, np.number):
        ts = df["timestamp"].astype(np.int64)
        df["timestamp"] = ts * 1000 if ts.max() < 10_000_000_000 else ts
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 1_000_000
    return df.sort_values("timestamp").reset_index(drop=True)

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def rolling_std(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).std(ddof=0)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
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

def simulate(df: pd.DataFrame, params: Dict, initial_balance: float, maker_fee: float, taker_fee: float):
    df = df.copy()
    close = df["close"].astype(float)

    bb_period = int(params.get("bb_period", 20))
    bb_std = float(params.get("bb_std", 2.0))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_long_max = float(params.get("rsi_long_max", 35))
    rsi_short_min = float(params.get("rsi_short_min", 65))
    adx_period = int(params.get("adx_period", 14))
    adx_max = float(params.get("adx_max", 18))
    atr_period = int(params.get("atr_period", 14))
    stop_atr_mult = float(params.get("stop_atr_mult", 1.2))

    leverage = float(params.get("leverage", 1))
    pos_pct = float(params.get("position_size_percentage", 20)) / 100.0
    use_longs = bool(params.get("use_longs", True))
    use_shorts = bool(params.get("use_shorts", True))

    basis = sma(close, bb_period)
    sd = rolling_std(close, bb_period)
    bb_up = basis + bb_std * sd
    bb_low = basis - bb_std * sd
    r = rsi(close, rsi_period)
    a = adx(df, adx_period)
    at = atr(df, atr_period)

    df["basis"]=basis; df["bb_up"]=bb_up; df["bb_low"]=bb_low; df["rsi"]=r; df["adx"]=a; df["atr"]=at

    balance = initial_balance
    pos_side: Optional[str] = None
    entry_px = 0.0
    qty = 0.0
    entry_atr = 0.0

    trades: List[Trade] = []
    equity_rows = []

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        row = df.iloc[i]
        ts = int(row["timestamp"])
        h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

        mtm = 0.0
        if pos_side == "long":
            mtm = (c-entry_px)*qty
        elif pos_side == "short":
            mtm = (entry_px-c)*qty
        equity_rows.append({"timestamp": ts, "balance": balance, "equity": balance+mtm, "pos_side": pos_side or "", "qty": qty})

        if any(prev[k]!=prev[k] for k in ["basis","bb_up","bb_low","rsi","adx","atr"]):
            continue

        if pos_side is not None:
            tp = float(prev["basis"])
            if pos_side == "long":
                sl = entry_px - stop_atr_mult * entry_atr
                if l <= sl <= h:
                    exit_reason="sl"; exit_price=sl
                elif l <= tp <= h:
                    exit_reason="tp"; exit_price=tp
                else:
                    continue
                fee = exit_price*qty*taker_fee
                pnl = (exit_price-entry_px)*qty - fee
                balance += (exit_price-entry_px)*qty
                balance -= fee
                trades.append(Trade("long", int(prev["timestamp"]), ts, entry_px, exit_price, qty, pnl, pnl/(entry_px*qty), exit_reason, fee))
                pos_side=None; entry_px=0.0; qty=0.0; entry_atr=0.0
            else:
                sl = entry_px + stop_atr_mult * entry_atr
                if l <= sl <= h:
                    exit_reason="sl"; exit_price=sl
                elif l <= tp <= h:
                    exit_reason="tp"; exit_price=tp
                else:
                    continue
                fee = exit_price*qty*taker_fee
                pnl = (entry_px-exit_price)*qty - fee
                balance += (entry_px-exit_price)*qty
                balance -= fee
                trades.append(Trade("short", int(prev["timestamp"]), ts, entry_px, exit_price, qty, pnl, pnl/(entry_px*qty), exit_reason, fee))
                pos_side=None; entry_px=0.0; qty=0.0; entry_atr=0.0
            continue

        if float(prev["adx"]) >= adx_max:
            continue

        notional = balance * pos_pct * leverage

        if use_longs and float(prev["close"]) <= float(prev["bb_low"]) and float(prev["rsi"]) <= rsi_long_max:
            entry = float(prev["bb_low"])
            if l <= entry <= h:
                q = notional / entry
                balance -= notional * maker_fee
                pos_side="long"; entry_px=entry; qty=q; entry_atr=float(prev["atr"])
        elif use_shorts and float(prev["close"]) >= float(prev["bb_up"]) and float(prev["rsi"]) >= rsi_short_min:
            entry = float(prev["bb_up"])
            if l <= entry <= h:
                q = notional / entry
                balance -= notional * maker_fee
                pos_side="short"; entry_px=entry; qty=q; entry_atr=float(prev["atr"])

    return pd.DataFrame([t.__dict__ for t in trades]), pd.DataFrame(equity_rows)

def summary(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
    if equity_df.empty:
        return {"error":"no equity"}
    start=float(equity_df["equity"].iloc[0])
    end=float(equity_df["equity"].iloc[-1])
    dd = equity_df["equity"].cummax() - equity_df["equity"]
    max_dd=float(dd.max()) if len(dd) else 0.0
    res={"start_equity":start,"end_equity":end,"return_pct":(end/start-1)*100 if start else 0.0,
         "max_drawdown":max_dd,"max_drawdown_pct":(max_dd/equity_df["equity"].cummax().max())*100 if len(dd) else 0.0,
         "num_trades":int(len(trades_df))}
    if not trades_df.empty:
        wins=trades_df[trades_df["pnl"]>0]; losses=trades_df[trades_df["pnl"]<=0]
        res["win_rate_pct"]=len(wins)/len(trades_df)*100
        res["avg_pnl"]=float(trades_df["pnl"].mean())
        res["profit_factor"]=float(wins["pnl"].sum()/abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum()!=0 else float("inf")
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bot-file", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--outdir", default="results")
    args=ap.parse_args()

    params=parse_params_from_bot_file(args.bot_file)
    symbol=params.get("symbol")
    timeframe=params.get("timeframe","1h")

    if args.csv:
        df=ensure_timestamp_ms(pd.read_csv(args.csv))
    else:
        if not args.start:
            raise ValueError("--start required when fetching via ccxt")
        df=fetch_ohlcv_ccxt(symbol, timeframe, to_ms(args.start), to_ms(args.end) if args.end else None, args.exchange)

    trades_df, equity_df = simulate(df, params, args.initial, args.maker_fee, args.taker_fee)

    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    name=Path(args.bot_file).stem
    trades_path=out/f"{name}_trades.csv"; equity_path=out/f"{name}_equity.csv"
    trades_df.to_csv(trades_path, index=False); equity_df.to_csv(equity_path, index=False)

    print(f"\n=== {name} ===")
    for k,v in summary(trades_df,equity_df).items():
        print(f"{k}: {v}")
    print(f"Saved -> {trades_path}")
    print(f"Saved -> {equity_path}")

if __name__=="__main__":
    main()
