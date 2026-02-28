#!/usr/bin/env python3
"""
backtest_portfolio_regime_switch_cached_v2.py

Fixes vs v1
- Robust trade timestamp detection + unit inference (ms vs s)
- Correctly reports ACTIVE trades per month (no more all-zeros when trades exist)
- Prints sanity stats: total trades (each strategy) and active trades after regime filter
- Keeps the same regime-switch portfolio equity construction (dynamic weights on returns)

Run example:
python3 code/strategies/envelope/backtest_portfolio_regime_switch_cached_v2.py \
  --btc-bot code/strategies/envelope/run_btc_trend_1h_v4.py \
  --sol-bot code/strategies/envelope/run_sol_bbrsi_1h_v3.py \
  --start "2025-02-28 00:00:00" \
  --end   "2026-02-28 00:00:00" \
  --initial 1000 \
  --name regime_v2 \
  --adx-period 14 --adx-on 27 --adx-off 23 \
  --vol-period 14 --vol-on 0.020 --vol-off 0.017 \
  --w-trend-btc 1.0 --w-range-btc 0.0
"""

from __future__ import annotations
import argparse, ast, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None


# ----------------- cache helpers -----------------
def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")

def _cache_path(cache_dir: str, exchange_id: str, symbol: str, timeframe: str) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    fname = f"{exchange_id}__{_sanitize_symbol(symbol)}__{timeframe}.csv.gz"
    return cache_root / fname

def _load_cached_ohlcv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        need = ["timestamp","open","high","low","close","volume"]
        for c in need:
            if c not in df.columns:
                return None
        df["timestamp"] = df["timestamp"].astype("int64")
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df[need]
    except Exception:
        return None

def _save_cached_ohlcv(path: Path, df: pd.DataFrame) -> None:
    out = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out.to_csv(path, index=False, compression="gzip")

def _merge_ohlcv(a: Optional[pd.DataFrame], b: pd.DataFrame) -> pd.DataFrame:
    if a is None or a.empty:
        return b.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out = pd.concat([a, b], ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out

def to_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_ohlcv_ccxt(symbol: str, timeframe: str, start_ms: int, end_ms: Optional[int], exchange_id: str = "bitget",
                     cache_dir: str = ".cache/ohlcv", use_cache: bool = True, refresh_if_no_end: bool = False) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Install with: pip install ccxt")

    cache_file = _cache_path(cache_dir, exchange_id, symbol, timeframe)
    cached = _load_cached_ohlcv(cache_file) if use_cache else None

    def _covered(df: pd.DataFrame, s: int, e: Optional[int]) -> bool:
        if df is None or df.empty:
            return False
        lo = int(df["timestamp"].min())
        hi = int(df["timestamp"].max())
        if e is None:
            return (lo <= s) and (not refresh_if_no_end)
        return (lo <= s) and (hi >= e)

    if cached is not None and _covered(cached, start_ms, end_ms):
        out = cached
    else:
        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        ex.load_markets()

        def _fetch_range(s_ms: int, e_ms: Optional[int]) -> pd.DataFrame:
            all_rows = []
            since = s_ms
            limit = 1000
            while True:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                if not batch:
                    break
                all_rows.extend(batch)
                last_ts = batch[-1][0]
                since = last_ts + 1
                if e_ms is not None and since > e_ms:
                    break
                if len(batch) >= 2 and batch[-1][0] == batch[-2][0]:
                    break
            df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = df["timestamp"].astype("int64")
            return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        fetched_parts: List[pd.DataFrame] = []
        if cached is None or cached.empty:
            fetched_parts.append(_fetch_range(start_ms, end_ms))
            out = fetched_parts[-1]
        else:
            lo = int(cached["timestamp"].min())
            hi = int(cached["timestamp"].max())
            if start_ms < lo:
                fetched_parts.append(_fetch_range(start_ms, lo))
            if end_ms is not None:
                if end_ms > hi:
                    fetched_parts.append(_fetch_range(hi, end_ms))
            else:
                if refresh_if_no_end:
                    fetched_parts.append(_fetch_range(hi, None))
            out = cached.copy()
            for part in fetched_parts:
                if part is not None and not part.empty:
                    out = _merge_ohlcv(out, part)

        if use_cache and out is not None and not out.empty:
            _save_cached_ohlcv(cache_file, out)

    if end_ms is None:
        out = out[out["timestamp"] >= start_ms].copy()
    else:
        out = out[(out["timestamp"] >= start_ms) & (out["timestamp"] <= end_ms)].copy()

    return out.reset_index(drop=True)


# ----------------- parse params -----------------
def parse_params_from_bot_file(bot_file: str) -> Dict[str, Any]:
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


# ----------------- dynamic module loading (Py3.14-safe) -----------------
def load_backtest_module(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("btmod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # Py3.14 dataclass fix
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# ----------------- indicators (ADX + ATR) -----------------
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


def build_regime(df_btc: pd.DataFrame, adx_period: int, adx_on: float, adx_off: float,
                 vol_period: int, vol_on: float, vol_off: float,
                 mode: str = "adx_and_vol") -> pd.DataFrame:
    out = df_btc[["timestamp","open","high","low","close","volume"]].copy()
    out["adx"] = compute_adx(out, adx_period)
    out["atr"] = compute_atr(out, vol_period)
    out["atrp"] = (out["atr"] / out["close"].astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    state = []
    cur = "RANGE"
    for i in range(len(out)):
        adx = float(out.loc[i, "adx"])
        atrp = float(out.loc[i, "atrp"])
        if mode == "adx_only":
            trend_on = adx >= adx_on
            trend_off = adx <= adx_off
        else:
            trend_on = (adx >= adx_on) and (atrp >= vol_on)
            trend_off = (adx <= adx_off) or (atrp <= vol_off)

        if cur == "RANGE" and trend_on:
            cur = "TREND"
        elif cur == "TREND" and trend_off:
            cur = "RANGE"
        state.append(cur)

    out["state"] = state
    return out[["timestamp","state","adx","atr","atrp"]]


# ----------------- performance helpers -----------------
def summarize_equity(equity: pd.Series) -> Dict[str, float]:
    eq = equity.astype(float).values
    start = float(eq[0])
    end = float(eq[-1])
    ret = (end / start - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max())
    max_dd_pct = float((max_dd / peak[np.argmax(dd)]) * 100.0) if max_dd > 0 else 0.0
    return {
        "start_equity": start,
        "end_equity": end,
        "return_pct": ret,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
    }

def monthly_summary(equity_df: pd.DataFrame, init_equity: float) -> pd.DataFrame:
    df = equity_df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["month"] = df["dt"].dt.strftime("%Y-%m")
    m = df.groupby("month")["equity"].last().to_frame("equity_end")
    m["equity_start"] = m["equity_end"].shift(1)
    m.iloc[0, m.columns.get_loc("equity_start")] = init_equity
    m["pnl"] = m["equity_end"] - m["equity_start"]
    m["ret_pct"] = (m["equity_end"] / m["equity_start"] - 1.0) * 100.0
    return m.reset_index()

# --------- trade timestamp normalization (FIX) ----------
def _find_trade_time_col(trades: pd.DataFrame) -> Optional[str]:
    if trades is None or trades.empty:
        return None
    candidates = [
        "entry_timestamp","entry_ts","open_timestamp","open_ts","timestamp","time",
        "entry_time","open_time","opened_at","entry_at","open_at",
        "datetime","date","created_at"
    ]
    for c in candidates:
        if c in trades.columns:
            return c
    # fallback: any column that looks like it contains timestamps
    for c in trades.columns:
        lc = str(c).lower()
        if "time" in lc or "date" in lc or "ts" in lc:
            return c
    return None

def _to_ts_ms(series: pd.Series) -> pd.Series:
    """Convert a trade time column to int64 epoch milliseconds with unit inference."""
    s = series.copy()

    # If already datetime-like
    if np.issubdtype(s.dtype, np.datetime64):
        return (pd.to_datetime(s, utc=True).astype("int64") // 1_000_000).astype("int64")

    # Try parse strings/datetimes
    if s.dtype == object:
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
        # if at least some parsed values exist, use them
        if parsed.notna().sum() > 0:
            return (parsed.astype("int64") // 1_000_000).astype("int64")

    # Numeric path
    num = pd.to_numeric(s, errors="coerce")
    # Infer unit by magnitude (median)
    med = float(num.dropna().median()) if num.dropna().shape[0] else 0.0
    if med > 1e12:     # already ms
        ts_ms = num
    elif med > 1e9:    # seconds (epoch)
        ts_ms = num * 1000.0
    else:
        # could be seconds with small values (unlikely) or index; treat as seconds
        ts_ms = num * 1000.0
    return ts_ms.round().astype("Int64").astype("int64")

def filter_trades_by_regime(trades: pd.DataFrame, regime_df: pd.DataFrame, active_state: str) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()

    tc = _find_trade_time_col(trades)
    if tc is None:
        return trades.copy()  # can't filter safely

    t = trades.copy()
    t["_ts"] = _to_ts_ms(t[tc])
    t = t.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

    r = regime_df[["timestamp","state"]].copy().sort_values("timestamp").rename(columns={"timestamp":"_ts"})
    t2 = pd.merge_asof(t, r, on="_ts", direction="backward")
    t2 = t2[t2["state"] == active_state].drop(columns=["_ts"]).reset_index(drop=True)
    return t2

def trades_monthly_counts(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["month","strategy","trades"])
    tc = _find_trade_time_col(trades_df)
    if tc is None:
        return pd.DataFrame(columns=["month","strategy","trades"])
    tmp = trades_df.copy()
    ts_ms = _to_ts_ms(tmp[tc])
    tmp["dt"] = pd.to_datetime(ts_ms, unit="ms", utc=True)
    tmp["month"] = tmp["dt"].dt.strftime("%Y-%m")
    if "strategy" not in tmp.columns:
        tmp["strategy"] = "UNKNOWN"
    out = tmp.groupby(["month","strategy"]).size().reset_index(name="trades")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-bot", required=True)
    ap.add_argument("--sol-bot", required=True)
    ap.add_argument("--backtest-module", default="code/strategies/envelope/backtest_portfolio_btc_sol_cached.py",
                    help="Backtest module that exposes simulate_trend and simulate_bbrsi")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--cache-dir", default=".cache/ohlcv")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")

    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--btc-leverage", type=float, default=None)
    ap.add_argument("--sol-leverage", type=float, default=None)

    # Regime params
    ap.add_argument("--mode", choices=["adx_only","adx_and_vol","adx_or_enter_and_exit"], default="adx_and_vol", help="Regime logic: adx_only; adx_and_vol (AND enter / OR exit); adx_or_enter_and_exit (OR enter / AND exit)")
    ap.add_argument("--adx-period", type=int, default=14)
    ap.add_argument("--adx-on", type=float, default=27.0)
    ap.add_argument("--adx-off", type=float, default=23.0)
    ap.add_argument("--vol-period", type=int, default=14)
    ap.add_argument("--vol-on", type=float, default=0.020)
    ap.add_argument("--vol-off", type=float, default=0.017)

    # Dynamic weights
    ap.add_argument("--w-trend-btc", type=float, default=1.0, help="BTC weight when state=TREND")
    ap.add_argument("--w-range-btc", type=float, default=0.0, help="BTC weight when state=RANGE")

    ap.add_argument("--outdir", default="results")
    ap.add_argument("--name", default="regime_switch")
    args = ap.parse_args()

    btc_base = parse_params_from_bot_file(args.btc_bot)
    sol_base = parse_params_from_bot_file(args.sol_bot)

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end) if args.end else None

    btc_df = fetch_ohlcv_ccxt(btc_base["symbol"], btc_base.get("timeframe","1h"), start_ms, end_ms,
                             exchange_id=args.exchange, cache_dir=args.cache_dir,
                             use_cache=(not args.no_cache), refresh_if_no_end=args.refresh_cache)
    sol_df = fetch_ohlcv_ccxt(sol_base["symbol"], sol_base.get("timeframe","1h"), start_ms, end_ms,
                             exchange_id=args.exchange, cache_dir=args.cache_dir,
                             use_cache=(not args.no_cache), refresh_if_no_end=args.refresh_cache)

    regime = build_regime(
        btc_df, args.adx_period, args.adx_on, args.adx_off,
        args.vol_period, args.vol_on, args.vol_off,
        mode=args.mode
    )

    bt = load_backtest_module(args.backtest_module)
    if not hasattr(bt, "simulate_trend") or not hasattr(bt, "simulate_bbrsi"):
        raise AttributeError("backtest module must expose simulate_trend and simulate_bbrsi")

    btc_init = args.initial
    sol_init = args.initial

    btc_trades, btc_eq = bt.simulate_trend(btc_df, btc_base, btc_init, args.maker_fee, args.taker_fee, args.btc_leverage)
    sol_trades, sol_eq = bt.simulate_bbrsi(sol_df, sol_base, sol_init, args.maker_fee, args.taker_fee, args.sol_leverage)

    # Convert trade logs to DF and tag
    btc_all = btc_trades.copy() if isinstance(btc_trades, pd.DataFrame) else pd.DataFrame(btc_trades)
    sol_all = sol_trades.copy() if isinstance(sol_trades, pd.DataFrame) else pd.DataFrame(sol_trades)
    if not btc_all.empty: btc_all["strategy"] = "BTC_TREND"
    if not sol_all.empty: sol_all["strategy"] = "SOL_BBRSI"

    # Filter trades to active regimes (FIXED)
    btc_active = filter_trades_by_regime(btc_all, regime, active_state="TREND")
    sol_active = filter_trades_by_regime(sol_all, regime, active_state="RANGE")
    trades_active_df = pd.concat([d for d in [btc_active, sol_active] if d is not None and not d.empty], ignore_index=True) if True else pd.DataFrame()
    trades_all_df = pd.concat([d for d in [btc_all, sol_all] if d is not None and not d.empty], ignore_index=True) if True else pd.DataFrame()

    # Portfolio equity via dynamic weights on returns
    b = btc_eq[["timestamp","equity"]].rename(columns={"equity":"equity_btc"}).sort_values("timestamp")
    s = sol_eq[["timestamp","equity"]].rename(columns={"equity":"equity_sol"}).sort_values("timestamp")
    df = pd.merge(b, s, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)
    df["equity_btc"] = df["equity_btc"].ffill().fillna(btc_init)
    df["equity_sol"] = df["equity_sol"].ffill().fillna(sol_init)
    df["ret_btc"] = df["equity_btc"].pct_change().fillna(0.0)
    df["ret_sol"] = df["equity_sol"].pct_change().fillna(0.0)
    df = pd.merge(df, regime[["timestamp","state","adx","atrp"]], on="timestamp", how="left")
    df["state"] = df["state"].ffill().fillna("RANGE")

    w_trend = float(args.w_trend_btc)
    w_range = float(args.w_range_btc)
    df["w_btc"] = np.where(df["state"] == "TREND", w_trend, w_range)
    df["w_sol"] = 1.0 - df["w_btc"]
    df["ret_port"] = df["w_btc"] * df["ret_btc"] + df["w_sol"] * df["ret_sol"]
    df["equity"] = args.initial * (1.0 + df["ret_port"]).cumprod()

    equity_df = df[["timestamp","equity","w_btc","w_sol","state","adx","atrp","equity_btc","equity_sol"]].copy()
    met = summarize_equity(equity_df["equity"])

    # Save outputs
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    name = args.name
    equity_path = outdir / f"portfolio_equity_{name}.csv"
    monthly_path = outdir / f"portfolio_monthly_{name}.csv"
    regime_path = outdir / f"portfolio_regime_{name}.csv"
    trades_all_path = outdir / f"portfolio_trades_all_{name}.csv"
    trades_active_path = outdir / f"portfolio_trades_filtered_{name}.csv"

    equity_df.to_csv(equity_path, index=False)
    monthly_df = monthly_summary(equity_df[["timestamp","equity"]], init_equity=args.initial)
    monthly_df.to_csv(monthly_path, index=False)
    regime.to_csv(regime_path, index=False)
    if not trades_all_df.empty: trades_all_df.to_csv(trades_all_path, index=False)
    if not trades_active_df.empty: trades_active_df.to_csv(trades_active_path, index=False)

    # Print summary
    print(f"\n=== Portfolio (Regime Switch v2): {name} ===")
    for k, v in met.items():
        print(f"{k}: {v}")

    print(f"\nTrades produced by simulators: BTC={len(btc_all)} SOL={len(sol_all)} TOTAL={len(trades_all_df)}")
    print(f"Trades ACTIVE (after regime filter): BTC={len(btc_active)} SOL={len(sol_active)} TOTAL={len(trades_active_df)}")

    # Monthly returns + active trades
    m = monthly_df.copy()
    counts = trades_monthly_counts(trades_active_df)
    if not counts.empty:
        pivot = counts.pivot(index="month", columns="strategy", values="trades").fillna(0).astype(int)
        pivot["trades_total"] = pivot.sum(axis=1)
        m = m.merge(pivot.reset_index(), on="month", how="left").fillna(0)

    print("\n=== Monthly returns (%) + ACTIVE trades ===")
    for _, r in m.iterrows():
        tr = int(r.get("trades_total", 0))
        btc_m = int(r.get("BTC_TREND", 0))
        sol_m = int(r.get("SOL_BBRSI", 0))
        print(f"{r['month']}: {r['ret_pct']:.2f}% (equity_end={r['equity_end']:.2f}) | trades={tr} (BTC={btc_m}, SOL={sol_m})")

    print(f"\nSaved -> {equity_path}")
    print(f"Saved -> {monthly_path}")
    print(f"Saved -> {regime_path}")
    if not trades_all_df.empty: print(f"Saved -> {trades_all_path}")
    if not trades_active_df.empty: print(f"Saved -> {trades_active_path}")

if __name__ == "__main__":
    main()
