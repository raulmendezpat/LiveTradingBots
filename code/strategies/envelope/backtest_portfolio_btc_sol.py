#!/usr/bin/env python3
"""
Portfolio backtest: BTC Trend (EMA pullback + ADX + ATR TP/SL) + SOL BBRSI (mean reversion)
- Loads params from bot files (AST parse `params = {...}`)
- Fetches OHLCV via ccxt (Bitget public)
- Runs each strategy on its own sub-balance (weights) and then combines equity curves
- Outputs:
  results/portfolio_trades_<name>.csv (combined trade log)
  results/portfolio_equity_<name>.csv (combined equity curve)
  results/portfolio_monthly_<name>.csv (month-by-month returns)
- Supports leverage override per bot to test 1x/2x/3x without editing bot files.
"""

from __future__ import annotations
import argparse, ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

# ----------------- helpers -----------------
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

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

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

# ----------------- trade record -----------------
@dataclass
class Trade:
    strategy: str
    symbol: str
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str

# ----------------- BBRSI simulation (single entry) -----------------
def simulate_bbrsi(df: pd.DataFrame, params: Dict, initial_balance: float, maker_fee: float, taker_fee: float, leverage_override: Optional[float]=None):
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
    if leverage_override is not None:
        leverage = float(leverage_override)

    pos_pct = float(params.get("position_size_percentage", 20)) / 100.0
    use_longs = bool(params.get("use_longs", True))
    use_shorts = bool(params.get("use_shorts", True))
    symbol = params.get("symbol", "")

    basis = sma(close, bb_period)
    sd = rolling_std(close, bb_period)
    bb_up = basis + bb_std * sd
    bb_low = basis - bb_std * sd
    rs = rsi(close, rsi_period)
    ax = adx(df, adx_period)
    at = atr(df, atr_period)

    df["basis"]=basis; df["bb_up"]=bb_up; df["bb_low"]=bb_low; df["rsi"]=rs; df["adx"]=ax; df["atr"]=at

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

        mtm=0.0
        if pos_side=="long":
            mtm=(c-entry_px)*qty
        elif pos_side=="short":
            mtm=(entry_px-c)*qty
        equity_rows.append({"timestamp": ts, "equity": balance+mtm, "balance": balance, "pos_side": pos_side or "", "qty": qty})

        if any(prev[k]!=prev[k] for k in ["basis","bb_up","bb_low","rsi","adx","atr"]):
            continue

        if pos_side is not None:
            tp = float(prev["basis"])
            if pos_side=="long":
                sl = entry_px - stop_atr_mult * entry_atr
                # conservative: SL first
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
                trades.append(Trade("bbrsi", symbol, "long", int(prev["timestamp"]), ts, entry_px, exit_price, qty, pnl, exit_reason))
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
                trades.append(Trade("bbrsi", symbol, "short", int(prev["timestamp"]), ts, entry_px, exit_price, qty, pnl, exit_reason))
            pos_side=None; entry_px=0.0; qty=0.0; entry_atr=0.0
            continue

        # no position
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

# ----------------- Trend simulation (EMA pullback) -----------------
def simulate_trend(df: pd.DataFrame, params: Dict, initial_balance: float, maker_fee: float, taker_fee: float, leverage_override: Optional[float]=None):
    df = df.copy()
    close = df["close"].astype(float)

    ema_fast_n = int(params.get("ema_fast", 20))
    ema_slow_n = int(params.get("ema_slow", 200))
    adx_period = int(params.get("adx_period", 14))
    adx_min = float(params.get("adx_min", 20))
    atr_period = int(params.get("atr_period", 14))
    stop_atr_mult = float(params.get("stop_atr_mult", 1.5))
    tp_atr_mult = float(params.get("tp_atr_mult", 2.0))

    leverage = float(params.get("leverage", 1))
    if leverage_override is not None:
        leverage = float(leverage_override)

    pos_pct = float(params.get("position_size_percentage", 20)) / 100.0
    use_longs = bool(params.get("use_longs", True))
    use_shorts = bool(params.get("use_shorts", True))
    symbol = params.get("symbol", "")

    df["ema_fast"]=ema(close, ema_fast_n)
    df["ema_slow"]=ema(close, ema_slow_n)
    df["adx"]=adx(df, adx_period)
    df["atr"]=atr(df, atr_period)

    balance = initial_balance
    pos_side: Optional[str]=None
    entry_px=0.0
    qty=0.0
    entry_atr=0.0

    trades: List[Trade]=[]
    equity_rows=[]

    for i in range(1, len(df)):
        prev=df.iloc[i-1]
        row=df.iloc[i]
        ts=int(row["timestamp"])
        h=float(row["high"]); l=float(row["low"]); c=float(row["close"])

        mtm=0.0
        if pos_side=="long":
            mtm=(c-entry_px)*qty
        elif pos_side=="short":
            mtm=(entry_px-c)*qty
        equity_rows.append({"timestamp": ts, "equity": balance+mtm, "balance": balance, "pos_side": pos_side or "", "qty": qty})

        if any(prev[k]!=prev[k] for k in ["ema_fast","ema_slow","adx","atr"]):
            continue

        if pos_side is not None:
            tp = entry_px + tp_atr_mult*entry_atr if pos_side=="long" else entry_px - tp_atr_mult*entry_atr
            sl = entry_px - stop_atr_mult*entry_atr if pos_side=="long" else entry_px + stop_atr_mult*entry_atr

            if l <= sl <= h:
                exit_reason="sl"; exit_price=sl
            elif l <= tp <= h:
                exit_reason="tp"; exit_price=tp
            else:
                continue

            fee = exit_price*qty*taker_fee
            if pos_side=="long":
                pnl=(exit_price-entry_px)*qty - fee
                balance += (exit_price-entry_px)*qty
                trades.append(Trade("trend", symbol, "long", int(prev["timestamp"]), ts, entry_px, exit_price, qty, pnl, exit_reason))
            else:
                pnl=(entry_px-exit_price)*qty - fee
                balance += (entry_px-exit_price)*qty
                trades.append(Trade("trend", symbol, "short", int(prev["timestamp"]), ts, entry_px, exit_price, qty, pnl, exit_reason))

            balance -= fee
            pos_side=None; entry_px=0.0; qty=0.0; entry_atr=0.0
            continue

        # no position: entry at EMA_fast if pullback and ADX strong
        if float(prev["adx"]) < adx_min:
            continue

        ema_fast_v=float(prev["ema_fast"])
        ema_slow_v=float(prev["ema_slow"])
        close_v=float(prev["close"])
        atr_v=float(prev["atr"])
        if atr_v!=atr_v or atr_v<=0:
            continue

        notional = balance * pos_pct * leverage

        if use_longs and ema_fast_v > ema_slow_v and close_v <= ema_fast_v:
            entry=ema_fast_v
            if l <= entry <= h:
                qty = notional / entry
                balance -= notional * maker_fee
                pos_side="long"; entry_px=entry; entry_atr=atr_v
        elif use_shorts and ema_fast_v < ema_slow_v and close_v >= ema_fast_v:
            entry=ema_fast_v
            if l <= entry <= h:
                qty = notional / entry
                balance -= notional * maker_fee
                pos_side="short"; entry_px=entry; entry_atr=atr_v

    return pd.DataFrame([t.__dict__ for t in trades]), pd.DataFrame(equity_rows)

# ----------------- portfolio metrics -----------------
def summarize_equity(equity: pd.Series) -> Dict:
    if equity.empty:
        return {"error":"no equity"}
    start=float(equity.iloc[0])
    end=float(equity.iloc[-1])
    dd = equity.cummax() - equity
    max_dd=float(dd.max()) if len(dd) else 0.0
    return {
        "start_equity": start,
        "end_equity": end,
        "return_pct": (end/start-1)*100 if start else 0.0,
        "max_drawdown": max_dd,
        "max_drawdown_pct": (max_dd/float(equity.cummax().max()))*100 if len(dd) else 0.0,
    }

def monthly_returns_from_equity(equity_df: pd.DataFrame) -> pd.DataFrame:
    tmp = equity_df.copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], unit="ms", utc=True)
    tmp["month"] = tmp["dt"].dt.strftime("%Y-%m")
    month_end = tmp.groupby("month", as_index=False).tail(1)[["month","equity"]].rename(columns={"equity":"equity_end"})
    month_end = month_end.sort_values("month").reset_index(drop=True)
    month_end["equity_start"] = month_end["equity_end"].shift(1)
    month_end.loc[0, "equity_start"] = float(tmp["equity"].iloc[0])
    month_end["return_pct"] = (month_end["equity_end"]/month_end["equity_start"] - 1)*100
    return month_end[["month","return_pct","equity_end"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-bot", required=True, help="BTC trend bot file (e.g., run_btc_trend_1h_v3.py)")
    ap.add_argument("--sol-bot", required=True, help="SOL BBRSI bot file (e.g., run_sol_bbrsi_1h_v3.py)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--initial", type=float, default=1000.0)

    ap.add_argument("--btc-weight", type=float, default=0.5, help="fraction of initial equity allocated to BTC bot")
    ap.add_argument("--sol-weight", type=float, default=0.5, help="fraction of initial equity allocated to SOL bot")

    ap.add_argument("--btc-leverage", type=float, default=None, help="override leverage for BTC bot (e.g., 1,2,3)")
    ap.add_argument("--sol-leverage", type=float, default=None, help="override leverage for SOL bot (e.g., 1,2,3)")

    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--name", default="btc_trend__sol_bbrsi", help="name suffix for output files")
    args = ap.parse_args()

    if abs((args.btc_weight + args.sol_weight) - 1.0) > 1e-6:
        raise ValueError("btc-weight + sol-weight must equal 1.0")

    btc_params = parse_params_from_bot_file(args.btc_bot)
    sol_params = parse_params_from_bot_file(args.sol_bot)

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end) if args.end else None

    # Fetch data per symbol
    btc_df = fetch_ohlcv_ccxt(btc_params["symbol"], btc_params.get("timeframe","1h"), start_ms, end_ms, args.exchange)
    sol_df = fetch_ohlcv_ccxt(sol_params["symbol"], sol_params.get("timeframe","1h"), start_ms, end_ms, args.exchange)

    btc_init = args.initial * args.btc_weight
    sol_init = args.initial * args.sol_weight

    btc_trades, btc_eq = simulate_trend(btc_df, btc_params, btc_init, args.maker_fee, args.taker_fee, args.btc_leverage)
    sol_trades, sol_eq = simulate_bbrsi(sol_df, sol_params, sol_init, args.maker_fee, args.taker_fee, args.sol_leverage)

    # Combine equity curves by timestamp (outer join, forward fill each, sum)
    btc_eq2 = btc_eq[["timestamp","equity"]].rename(columns={"equity":"equity_btc"})
    sol_eq2 = sol_eq[["timestamp","equity"]].rename(columns={"equity":"equity_sol"})
    comb = pd.merge(btc_eq2, sol_eq2, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)
    comb["equity_btc"] = comb["equity_btc"].ffill()
    comb["equity_sol"] = comb["equity_sol"].ffill()
    # Fill initial gaps with initial allocations
    comb["equity_btc"] = comb["equity_btc"].fillna(btc_init)
    comb["equity_sol"] = comb["equity_sol"].fillna(sol_init)
    comb["equity"] = comb["equity_btc"] + comb["equity_sol"]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    trades_all = pd.concat([btc_trades, sol_trades], ignore_index=True)
    trades_all = trades_all.sort_values(["exit_ts","strategy"]).reset_index(drop=True)

    trades_path = out / f"portfolio_trades_{args.name}.csv"
    equity_path = out / f"portfolio_equity_{args.name}.csv"
    monthly_path = out / f"portfolio_monthly_{args.name}.csv"

    trades_all.to_csv(trades_path, index=False)
    comb.to_csv(equity_path, index=False)
    monthly_df = monthly_returns_from_equity(comb[["timestamp","equity"]])
    monthly_df.to_csv(monthly_path, index=False)

    print(f"\n=== Portfolio: {args.name} ===")
    for k,v in summarize_equity(comb["equity"]).items():
        print(f"{k}: {v}")

    print("\n=== Monthly returns (%) ===")
    for _, r in monthly_df.iterrows():
        print(f"{r['month']}: {r['return_pct']:.2f}% (equity_end={r['equity_end']:.2f})")

    print(f"\nSaved -> {trades_path}")
    print(f"Saved -> {equity_path}")
    print(f"Saved -> {monthly_path}")

if __name__ == "__main__":
    main()
