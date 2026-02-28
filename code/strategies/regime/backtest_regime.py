#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

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
                    val = ast.literal_eval(node.value)
                    if not isinstance(val, dict):
                        raise ValueError("params is not a dict")
                    return val
    raise ValueError(f"Could not find params dict in {bot_file}")


def to_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_ohlcv_ccxt(symbol: str, timeframe: str, start_ms: int, end_ms: Optional[int], exchange_id: str) -> pd.DataFrame:
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
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if end_ms is not None:
        df = df[df["timestamp"] <= end_ms].reset_index(drop=True)
    return df


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rolling_std(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).std(ddof=0)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def annualized_vol_1h(close: pd.Series, lookback_hours: int) -> pd.Series:
    r = np.log(close).diff()
    hv = r.rolling(lookback_hours).std(ddof=0)
    return hv * np.sqrt(24 * 365)


def donchian(df: pd.DataFrame, n: int) -> Tuple[pd.Series, pd.Series]:
    high_n = df["high"].rolling(n).max()
    low_n = df["low"].rolling(n).min()
    return high_n, low_n


def monthly_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame(columns=["month", "return_pct", "equity_end"])
    tmp = equity_df.copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], unit="ms", utc=True)
    tmp["month"] = tmp["dt"].dt.strftime("%Y-%m")
    month_end = tmp.groupby("month", as_index=False).tail(1)[["month", "equity"]].rename(columns={"equity": "equity_end"})
    month_end = month_end.sort_values("month").reset_index(drop=True)
    month_end["equity_start"] = month_end["equity_end"].shift(1)
    month_end.loc[0, "equity_start"] = float(tmp["equity"].iloc[0])
    month_end["return_pct"] = (month_end["equity_end"] / month_end["equity_start"] - 1) * 100
    return month_end[["month", "return_pct", "equity_end"]]


def summarize(trades: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
    eq = equity_df["equity"].astype(float)
    start = float(eq.iloc[0]) if len(eq) else np.nan
    end = float(eq.iloc[-1]) if len(eq) else np.nan
    peak = eq.cummax()
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0
    peak_at = float(peak.iloc[dd.idxmax()]) if len(dd) else float(peak.iloc[-1]) if len(eq) else 0.0
    max_dd_pct = (max_dd / peak_at * 100.0) if peak_at > 0 else 0.0

    if trades.empty:
        return dict(
            start_equity=start,
            end_equity=end,
            return_pct=(end / start - 1) * 100 if start else np.nan,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            num_trades=0,
            win_rate_pct=np.nan,
            avg_pnl=np.nan,
            profit_factor=np.nan,
        )

    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = (wins / losses) if losses > 0 else np.inf
    win_rate = (pnl > 0).mean() * 100.0

    return dict(
        start_equity=start,
        end_equity=end,
        return_pct=(end / start - 1) * 100 if start else np.nan,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        num_trades=int(len(trades)),
        win_rate_pct=float(win_rate),
        avg_pnl=float(pnl.mean()),
        profit_factor=float(pf),
    )


@dataclass
class Position:
    side: str       # "long" or "short"
    entry_px: float
    qty: float
    trail_stop: float
    mode: str       # "trend" or "mr"
    entry_atr: float


def calc_leverage(params: Dict, vol_ann: float, regime_boost: float) -> float:
    base = float(params.get("base_leverage", 1.0))
    target_vol = float(params.get("target_vol_annual", 1.0))
    min_lev = float(params.get("min_leverage", 1.0))
    max_lev = float(params.get("max_leverage", 5.0))
    if vol_ann != vol_ann or vol_ann <= 0:
        lev = base
    else:
        lev = base * (target_vol / vol_ann)
    lev *= float(regime_boost)
    return float(np.clip(lev, min_lev, max_lev))


def run_backtest(df: pd.DataFrame, params: Dict, initial: float, maker_fee: float, taker_fee: float):
    df = df.copy()
    close = df["close"].astype(float)

    df["ema_fast"] = ema(close, int(params["ema_fast"]))
    df["ema_slow"] = ema(close, int(params["ema_slow"]))
    df["adx"] = adx(df, int(params.get("adx_period", 14)))
    df["atr"] = atr(df, int(params["atr_period"]))
    df["vol_ann"] = annualized_vol_1h(close, int(params.get("vol_lookback_hours", 168)))

    dc_hi, dc_lo = donchian(df, int(params["donchian_period"]))
    df["dc_hi"] = dc_hi
    df["dc_lo"] = dc_lo

    bb_p = int(params["bb_period"])
    bb_k = float(params["bb_std"])
    basis = sma(close, bb_p)
    sd = rolling_std(close, bb_p)
    df["bb_basis"] = basis
    df["bb_up"] = basis + bb_k * sd
    df["bb_low"] = basis - bb_k * sd
    df["rsi"] = rsi(close, int(params["rsi_period"]))

    adx_trend_min = float(params.get("adx_trend_min", 18))
    adx_chop_max = float(params.get("adx_chop_max", 16))
    confirm = float(params.get("breakout_confirm", 0.0))
    trail_mult = float(params["trail_atr_mult"])
    mr_sl_mult = float(params["mr_stop_atr_mult"])
    rsi_long_max = float(params["rsi_long_max"])
    rsi_short_min = float(params["rsi_short_min"])
    pos_pct = float(params["position_size_percentage"]) / 100.0
    use_longs = bool(params.get("use_longs", True))
    use_shorts = bool(params.get("use_shorts", True))
    cd_bars = int(params.get("cooldown_bars_after_exit", 0))
    trend_boost = float(params.get("trend_leverage_boost", 1.0))
    mr_boost = float(params.get("mr_leverage_boost", 1.0))

    balance = float(initial)
    pos: Optional[Position] = None
    cooldown = 0
    trades = []
    equity_rows = []

    def mark_to_market(c: float) -> float:
        if pos is None:
            return 0.0
        if pos.side == "long":
            return (c - pos.entry_px) * pos.qty
        return (pos.entry_px - c) * pos.qty

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        row = df.iloc[i]
        ts = int(row["timestamp"])
        h = float(row["high"]); l = float(row["low"]); c = float(row["close"])

        mtm = mark_to_market(c)
        equity_rows.append({
            "timestamp": ts,
            "equity": balance + mtm,
            "balance": balance,
            "pos_side": pos.side if pos else "",
            "pos_mode": pos.mode if pos else "",
            "qty": pos.qty if pos else 0.0
        })

        needed = ["ema_fast", "ema_slow", "adx", "atr", "vol_ann", "dc_hi", "dc_lo", "bb_basis", "bb_up", "bb_low", "rsi"]
        if any(prev[k] != prev[k] for k in needed):
            continue

        if cooldown > 0:
            cooldown -= 1

        # EXIT
        if pos is not None:
            exit_reason = None
            exit_price = None

            if pos.mode == "trend":
                if pos.side == "long":
                    new_trail = float(prev["close"]) - trail_mult * float(prev["atr"])
                    pos.trail_stop = max(pos.trail_stop, new_trail)
                    if l <= pos.trail_stop <= h:
                        exit_reason = "trail_sl"
                        exit_price = pos.trail_stop
                    elif use_shorts and float(prev["close"]) < float(prev["dc_lo"]) * (1 - confirm):
                        exit_reason = "opp_break"
                        exit_price = float(prev["close"])
                else:
                    new_trail = float(prev["close"]) + trail_mult * float(prev["atr"])
                    pos.trail_stop = min(pos.trail_stop, new_trail)
                    if l <= pos.trail_stop <= h:
                        exit_reason = "trail_sl"
                        exit_price = pos.trail_stop
                    elif use_longs and float(prev["close"]) > float(prev["dc_hi"]) * (1 + confirm):
                        exit_reason = "opp_break"
                        exit_price = float(prev["close"])
            else:
                tp = float(prev["bb_basis"])
                if pos.side == "long":
                    sl = pos.entry_px - mr_sl_mult * pos.entry_atr
                    if l <= sl <= h:
                        exit_reason = "sl"; exit_price = sl
                    elif l <= tp <= h:
                        exit_reason = "tp"; exit_price = tp
                else:
                    sl = pos.entry_px + mr_sl_mult * pos.entry_atr
                    if l <= sl <= h:
                        exit_reason = "sl"; exit_price = sl
                    elif l <= tp <= h:
                        exit_reason = "tp"; exit_price = tp

            if exit_reason is not None and exit_price is not None:
                fee = exit_price * pos.qty * taker_fee
                pnl_gross = (exit_price - pos.entry_px) * pos.qty if pos.side == "long" else (pos.entry_px - exit_price) * pos.qty
                pnl = pnl_gross - fee
                balance += pnl_gross
                balance -= fee
                trades.append({
                    "side": pos.side,
                    "mode": pos.mode,
                    "entry_ts": int(prev["timestamp"]),
                    "exit_ts": ts,
                    "entry_price": pos.entry_px,
                    "exit_price": exit_price,
                    "qty": pos.qty,
                    "pnl": pnl,
                    "reason": exit_reason,
                })
                pos = None
                cooldown = cd_bars
                continue

        # ENTRY
        if pos is None and cooldown == 0:
            is_trend = float(prev["adx"]) >= adx_trend_min
            allow_mr = float(prev["adx"]) <= adx_chop_max

            vol_ann = float(prev["vol_ann"])
            regime_boost = trend_boost if is_trend else (mr_boost if allow_mr else 1.0)
            lev = calc_leverage(params, vol_ann, regime_boost)

            notional = balance * pos_pct * lev
            if notional <= 0:
                continue

            ef = float(prev["ema_fast"]); es = float(prev["ema_slow"])
            dc_h = float(prev["dc_hi"]); dc_l = float(prev["dc_lo"])
            atr_v = float(prev["atr"])
            if atr_v != atr_v or atr_v <= 0:
                continue

            if is_trend:
                if use_longs and ef > es and float(prev["close"]) > dc_h * (1 + confirm):
                    entry = dc_h * (1 + confirm)
                    if l <= entry <= h:
                        qty = notional / entry
                        balance -= notional * maker_fee
                        trail = entry - trail_mult * atr_v
                        pos = Position("long", entry, qty, trail, "trend", atr_v)
                        continue
                if use_shorts and ef < es and float(prev["close"]) < dc_l * (1 - confirm):
                    entry = dc_l * (1 - confirm)
                    if l <= entry <= h:
                        qty = notional / entry
                        balance -= notional * maker_fee
                        trail = entry + trail_mult * atr_v
                        pos = Position("short", entry, qty, trail, "trend", atr_v)
                        continue

            if allow_mr:
                bb_low = float(prev["bb_low"]); bb_up = float(prev["bb_up"])
                rsi_v = float(prev["rsi"])
                if use_longs and float(prev["close"]) <= bb_low and rsi_v <= rsi_long_max:
                    entry = bb_low
                    if l <= entry <= h:
                        qty = notional / entry
                        balance -= notional * maker_fee
                        pos = Position("long", entry, qty, entry - mr_sl_mult * atr_v, "mr", atr_v)
                        continue
                if use_shorts and float(prev["close"]) >= bb_up and rsi_v >= rsi_short_min:
                    entry = bb_up
                    if l <= entry <= h:
                        qty = notional / entry
                        balance -= notional * maker_fee
                        pos = Position("short", entry, qty, entry + mr_sl_mult * atr_v, "mr", atr_v)
                        continue

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    monthly_df = monthly_returns(equity_df)
    return trades_df, equity_df, monthly_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-file", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--exchange", default=None)
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--maker-fee", type=float, default=None)
    ap.add_argument("--taker-fee", type=float, default=None)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    params = parse_params_from_bot_file(args.bot_file)
    name = params.get("name", Path(args.bot_file).stem)

    exchange = args.exchange or params.get("exchange", "bitget")
    maker_fee = float(args.maker_fee) if args.maker_fee is not None else float(params.get("maker_fee", 0.0002))
    taker_fee = float(args.taker_fee) if args.taker_fee is not None else float(params.get("taker_fee", 0.0006))

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end) if args.end else None

    df = fetch_ohlcv_ccxt(params["symbol"], params.get("timeframe", "1h"), start_ms, end_ms, exchange)
    trades_df, equity_df, monthly_df = run_backtest(df, params, args.initial, maker_fee, taker_fee)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    trades_path = out / f"{name}_trades.csv"
    equity_path = out / f"{name}_equity.csv"
    monthly_path = out / f"{name}_monthly.csv"

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)

    s = summarize(trades_df, equity_df)
    print(f"\n=== {name} ===")
    for k, v in s.items():
        print(f"{k}: {v}")

    print("\n=== Monthly returns (%) ===")
    for _, r in monthly_df.iterrows():
        print(f"{r['month']}: {r['return_pct']:.2f}% (equity_end={r['equity_end']:.2f})")

    print(f"\nSaved -> {trades_path}")
    print(f"Saved -> {equity_path}")
    print(f"Saved -> {monthly_path}")


if __name__ == "__main__":
    main()
