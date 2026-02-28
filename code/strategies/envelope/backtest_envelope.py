#!/usr/bin/env python3
"""
Envelope strategy backtester (Bitget-style bots).

Features
- Loads params from an existing bot python file WITHOUT executing it (AST parse of `params = {...}`).
- Fetches OHLCV via ccxt (public data) OR reads OHLCV from CSV.
- Simulates multi-level envelope entries (3 levels etc.), trend directional filter w/ buffer, stop-loss, and mean-reversion take-profit to average.
- Outputs trades + equity curve CSV and prints performance summary.

Usage examples
1) Backtest using ccxt (Bitget public candles):
   python backtest_envelope.py --bot-file code/strategies/envelope/run_btc_1h_both.py --start 2025-01-01 --end 2026-02-28

2) Backtest from CSV (must contain: timestamp, open, high, low, close, volume; timestamp in ms or ISO8601):
   python backtest_envelope.py --bot-file run_btc_1h_both.py --csv BTCUSDT_1h.csv

Outputs
- results/<botname>_trades.csv
- results/<botname>_equity.csv
"""

from __future__ import annotations

import argparse
import ast
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dependency: ccxt for fetching OHLCV
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

# ----------------------------
# Utilities
# ----------------------------

def parse_params_from_bot_file(bot_file: str) -> Dict:
    """
    Parse a dict assigned to a top-level variable named `params` in a .py file using AST.
    This avoids executing the strategy script.
    """
    src = Path(bot_file).read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src, filename=bot_file)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "params":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise ValueError("Found params assignment but it is not a dict.")
                    return value
    raise ValueError(f"Could not find a top-level `params = {{...}}` dict in {bot_file}")


def to_ms(ts: str) -> int:
    # Accept YYYY-MM-DD or full ISO; return milliseconds since epoch UTC.
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ensure_timestamp_ms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column.")
    # If timestamp is numeric-like, treat as ms or s
    if np.issubdtype(df["timestamp"].dtype, np.number):
        ts = df["timestamp"].astype(np.int64)
        # Heuristic: if in seconds, convert to ms
        if ts.max() < 10_000_000_000:  # < year 2286 in seconds
            df["timestamp"] = ts * 1000
        else:
            df["timestamp"] = ts
    else:
        # Parse ISO-like
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 1_000_000
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Lightweight ADX implementation for optional regime analysis.
    """
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.rolling(period).mean()


# ----------------------------
# Backtest core
# ----------------------------

@dataclass
class Trade:
    side: str  # "long" or "short"
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    reason: str  # "tp" or "sl"
    fees: float


def compute_envelopes(avg: pd.Series, envelopes: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (band_low_df, band_high_df) each with columns band_1..band_n
    """
    lows = {}
    highs = {}
    for i, e in enumerate(envelopes, start=1):
        lows[f"band_{i}"] = avg * (1 - e)
        highs[f"band_{i}"] = avg * (1 + e)
    return pd.DataFrame(lows), pd.DataFrame(highs)


def trend_permissions(price: float, trend_ema: float, buffer_pct: float, directional: bool) -> Tuple[bool, bool]:
    """
    Returns (allow_long, allow_short)
    """
    if math.isnan(price) or math.isnan(trend_ema):
        return False, False

    if not directional:
        # If not directional, allow both (caller may implement legacy avg-vs-trend logic)
        return True, True

    upper = trend_ema * (1 + buffer_pct)
    lower = trend_ema * (1 - buffer_pct)

    if price > upper:
        return True, False
    if price < lower:
        return False, True
    return False, False


def simulate_envelope(
    df: pd.DataFrame,
    params: Dict,
    initial_balance: float = 1000.0,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0006,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Trade]]:
    """
    Simulate one symbol with envelope mean-reversion:
    - Enter long via limit at band_low levels; TP at avg; SL at avg_entry*(1-stop_loss_pct)
    - Enter short via limit at band_high levels; TP at avg; SL at avg_entry*(1+stop_loss_pct)
    Trend filter: price vs EMA(trend_filter_period) with optional buffer.
    Uses last CLOSED candle as signal; orders filled within next candle if touched.
    """
    df = df.copy()

    # Basic validation
    for col in ["timestamp", "open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")

    timeframe = params.get("timeframe", "1h")
    avg_period = int(params.get("average_period", 20))
    avg_type = params.get("average_type", "EMA").upper()
    envelopes = list(params.get("envelopes", [0.01, 0.015, 0.02]))
    stop_loss_pct = float(params.get("stop_loss_pct", 0.02))
    leverage = float(params.get("leverage", 1))
    pos_pct = float(params.get("position_size_percentage", 30.0)) / 100.0
    use_longs = bool(params.get("use_longs", True))
    use_shorts = bool(params.get("use_shorts", True))

    trend_filter = bool(params.get("trend_filter", False))
    trend_period = int(params.get("trend_filter_period", 100))
    trend_buffer_pct = float(params.get("trend_buffer_pct", 0.0))
    trend_directional = bool(params.get("trend_filter_directional", True))

    max_extension_pct = params.get("max_extension_pct", None)  # apply ONLY to shorts (recommended)
    if max_extension_pct is not None:
        max_extension_pct = float(max_extension_pct)

    # Indicators
    close = df["close"].astype(float)
    if avg_type == "EMA":
        avg = ema(close, avg_period)
    else:
        avg = close.rolling(avg_period).mean()  # SMA fallback
    trend_ema = ema(close, trend_period)

    band_low, band_high = compute_envelopes(avg, envelopes)

    df["avg"] = avg
    df["trend"] = trend_ema
    for i in range(1, len(envelopes) + 1):
        df[f"low_{i}"] = band_low[f"band_{i}"]
        df[f"high_{i}"] = band_high[f"band_{i}"]

    # State
    balance = initial_balance
    equity = initial_balance
    pos_side: Optional[str] = None  # "long"/"short"
    pos_qty = 0.0
    pos_entry_value = 0.0  # sum(price*qty) for avg entry
    filled_levels = set()

    trades: List[Trade] = []
    equity_rows = []

    n_levels = len(envelopes)
    # Notional per "full idea" is balance * pos_pct * leverage; split across levels.
    def level_notional(current_balance: float) -> float:
        return (current_balance * pos_pct * leverage) / n_levels

    def avg_entry_price() -> float:
        return pos_entry_value / pos_qty if pos_qty > 0 else float("nan")

    # Iterate candles; use signal from i-1 to trade candle i
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        row = df.iloc[i]

        ts = int(row["timestamp"])
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

        # mark-to-market equity
        mtm = 0.0
        if pos_side == "long" and pos_qty > 0:
            mtm = (c - avg_entry_price()) * pos_qty
        elif pos_side == "short" and pos_qty > 0:
            mtm = (avg_entry_price() - c) * pos_qty
        equity = balance + mtm
        equity_rows.append({"timestamp": ts, "balance": balance, "equity": equity, "pos_side": pos_side or "", "pos_qty": pos_qty})

        # Skip until indicators valid
        if math.isnan(float(prev["avg"])) or math.isnan(float(prev["trend"])):
            continue

        # Determine permissions from prev CLOSED candle
        prev_price = float(prev["close"])
        prev_avg = float(prev["avg"])
        prev_trend = float(prev["trend"])
        allow_long, allow_short = (True, True)

        if trend_filter:
            allow_long, allow_short = trend_permissions(prev_price, prev_trend, trend_buffer_pct, trend_directional)

        # Anti-momentum extension: apply only to SHORTs (distance above avg)
        if allow_short and (max_extension_pct is not None):
            if prev_price > prev_avg * (1 + max_extension_pct):
                allow_short = False

        # If no position: try to open via levels
        if pos_side is None:
            filled_levels.clear()

            # Long entries (limits at low bands)
            if use_longs and allow_long:
                for lvl in range(1, n_levels + 1):
                    entry = float(prev[f"low_{lvl}"])
                    if l <= entry <= h:
                        notional = level_notional(balance)
                        qty = notional / entry
                        fee = notional * maker_fee
                        balance -= fee
                        pos_side = "long"
                        pos_qty += qty
                        pos_entry_value += entry * qty
                        filled_levels.add(lvl)

            # Short entries (limits at high bands)
            if pos_side is None and use_shorts and allow_short:
                for lvl in range(1, n_levels + 1):
                    entry = float(prev[f"high_{lvl}"])
                    if l <= entry <= h:
                        notional = level_notional(balance)
                        qty = notional / entry
                        fee = notional * maker_fee
                        balance -= fee
                        pos_side = "short"
                        pos_qty += qty
                        pos_entry_value += entry * qty
                        filled_levels.add(lvl)

        else:
            # If position open, you may add more levels if price reaches deeper bands (same direction only)
            if pos_side == "long":
                # Add remaining levels
                if use_longs and allow_long:
                    for lvl in range(1, n_levels + 1):
                        if lvl in filled_levels:
                            continue
                        entry = float(prev[f"low_{lvl}"])
                        if l <= entry <= h:
                            notional = level_notional(balance)
                            qty = notional / entry
                            fee = notional * maker_fee
                            balance -= fee
                            pos_qty += qty
                            pos_entry_value += entry * qty
                            filled_levels.add(lvl)

                # Exit logic: TP at prev_avg; SL at avg_entry*(1-stop_loss_pct)
                entry_px = avg_entry_price()
                tp = prev_avg
                sl = entry_px * (1 - stop_loss_pct)

                exit_reason = None
                exit_px = None
                # If both TP and SL occur in same candle, assume worst (SL first) for conservatism.
                if l <= sl <= h:
                    exit_reason = "sl"
                    exit_px = sl
                elif l <= tp <= h:
                    exit_reason = "tp"
                    exit_px = tp

                if exit_reason and exit_px is not None:
                    notional_exit = exit_px * pos_qty
                    fee = notional_exit * taker_fee
                    pnl = (exit_px - entry_px) * pos_qty - fee
                    pnl_pct = pnl / (entry_px * pos_qty) if entry_px > 0 else 0.0
                    balance += (exit_px - entry_px) * pos_qty
                    balance -= fee

                    trades.append(Trade(
                        side="long",
                        entry_ts=int(df.loc[df.index[i-1], "timestamp"]),  # approximate
                        exit_ts=ts,
                        entry_price=entry_px,
                        exit_price=exit_px,
                        qty=pos_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        reason=exit_reason,
                        fees=fee,
                    ))
                    pos_side = None
                    pos_qty = 0.0
                    pos_entry_value = 0.0
                    filled_levels.clear()

            elif pos_side == "short":
                if use_shorts and allow_short:
                    for lvl in range(1, n_levels + 1):
                        if lvl in filled_levels:
                            continue
                        entry = float(prev[f"high_{lvl}"])
                        if l <= entry <= h:
                            notional = level_notional(balance)
                            qty = notional / entry
                            fee = notional * maker_fee
                            balance -= fee
                            pos_qty += qty
                            pos_entry_value += entry * qty
                            filled_levels.add(lvl)

                entry_px = avg_entry_price()
                tp = prev_avg
                sl = entry_px * (1 + stop_loss_pct)

                exit_reason = None
                exit_px = None
                # Conservative: SL first if both touched.
                if l <= sl <= h:
                    exit_reason = "sl"
                    exit_px = sl
                elif l <= tp <= h:
                    exit_reason = "tp"
                    exit_px = tp

                if exit_reason and exit_px is not None:
                    notional_exit = exit_px * pos_qty
                    fee = notional_exit * taker_fee
                    pnl = (entry_px - exit_px) * pos_qty - fee
                    pnl_pct = pnl / (entry_px * pos_qty) if entry_px > 0 else 0.0
                    balance += (entry_px - exit_px) * pos_qty
                    balance -= fee

                    trades.append(Trade(
                        side="short",
                        entry_ts=int(df.loc[df.index[i-1], "timestamp"]),
                        exit_ts=ts,
                        entry_price=entry_px,
                        exit_price=exit_px,
                        qty=pos_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        reason=exit_reason,
                        fees=fee,
                    ))
                    pos_side = None
                    pos_qty = 0.0
                    pos_entry_value = 0.0
                    filled_levels.clear()

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, equity_df, trades


def fetch_ohlcv_ccxt(symbol: str, timeframe: str, start_ms: int, end_ms: Optional[int], exchange_id: str = "bitget") -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Install with: pip install ccxt")

    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})

    # Bitget usually supports ~200 candles per fetch; we paginate.
    limit = 200
    all_rows = []
    since = start_ms

    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        # Avoid infinite loop
        if last_ts == since:
            break
        since = last_ts + 1
        if end_ms is not None and since >= end_ms:
            break

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if end_ms is not None:
        df = df[df["timestamp"] <= end_ms].reset_index(drop=True)
    return df


def summary(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
    if equity_df.empty:
        return {"error": "No equity data."}

    start_equity = float(equity_df["equity"].iloc[0])
    end_equity = float(equity_df["equity"].iloc[-1])

    dd = equity_df["equity"].cummax() - equity_df["equity"]
    max_dd = float(dd.max()) if len(dd) else 0.0
    max_dd_pct = max_dd / float(equity_df["equity"].cummax().max()) if len(dd) else 0.0

    res = {
        "start_equity": start_equity,
        "end_equity": end_equity,
        "return_pct": (end_equity / start_equity - 1) * 100 if start_equity else 0.0,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct * 100,
        "num_trades": int(len(trades_df)),
    }

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        res.update({
            "win_rate_pct": (len(wins) / len(trades_df)) * 100,
            "avg_pnl": float(trades_df["pnl"].mean()),
            "profit_factor": float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else float("inf"),
        })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-file", required=True, help="Path to run_btc_1h_both.py / run_sol_1h_both.py (contains `params = {...}`)")
    ap.add_argument("--csv", default=None, help="Optional OHLCV CSV instead of fetching via ccxt.")
    ap.add_argument("--exchange", default="bitget", help="ccxt exchange id (default: bitget)")
    ap.add_argument("--start", default=None, help="Start date ISO (e.g. 2025-01-01). Required if fetching.")
    ap.add_argument("--end", default=None, help="End date ISO (e.g. 2026-02-28). Optional.")
    ap.add_argument("--initial", type=float, default=1000.0, help="Initial balance in USDT")
    ap.add_argument("--maker-fee", type=float, default=0.0002, help="Maker fee rate (e.g. 0.0002 = 0.02%)")
    ap.add_argument("--taker-fee", type=float, default=0.0006, help="Taker fee rate (e.g. 0.0006 = 0.06%)")
    ap.add_argument("--outdir", default="results", help="Output directory")
    args = ap.parse_args()

    params = parse_params_from_bot_file(args.bot_file)
    symbol = params.get("symbol")
    timeframe = params.get("timeframe", "1h")

    if args.csv:
        df = pd.read_csv(args.csv)
        df = ensure_timestamp_ms(df)
    else:
        if not symbol:
            raise ValueError("params['symbol'] is missing.")
        if not args.start:
            raise ValueError("--start is required when fetching via ccxt.")
        start_ms = to_ms(args.start)
        end_ms = to_ms(args.end) if args.end else None
        df = fetch_ohlcv_ccxt(symbol, timeframe, start_ms, end_ms, args.exchange)

    trades_df, equity_df, _ = simulate_envelope(
        df=df,
        params=params,
        initial_balance=args.initial,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    botname = Path(args.bot_file).stem

    trades_path = outdir / f"{botname}_trades.csv"
    equity_path = outdir / f"{botname}_equity.csv"
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    res = summary(trades_df, equity_df)
    print("\n=== Backtest Summary ===")
    for k, v in res.items():
        print(f"{k}: {v}")
    print(f"\nSaved trades -> {trades_path}")
    print(f"Saved equity -> {equity_path}")


if __name__ == "__main__":
    main()
