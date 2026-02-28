#!/usr/bin/env python3
"""
backtest_portfolio_regime_switch_cached.py

Portfolio backtest with *regime switching* (trend vs mean-reversion) using OHLCV disk cache.

Why this exists
- Your walk-forward optimisation showed classic regime overfit.
- Instead of over-optimising parameters, we introduce a structural edge:
    * Use BTC Trend strategy ONLY when market is trending.
    * Use SOL Mean-Reversion (BB+RSI) ONLY when market is ranging / low trend.

How it works (high level)
1) Load baseline bot params from the bot files (expects `params = {...}` dict).
2) Fetch OHLCV once per symbol (cache in ./.cache/ohlcv).
3) Run both strategy simulators independently to get:
   - equity curves per strategy
   - trade logs per strategy
4) Build a regime signal from BTC OHLCV:
   - ADX (trend strength)
   - ATR/Close (volatility proxy)
   - Optional hysteresis to reduce switching noise
5) Combine strategy returns using dynamic weights:
   - In TREND regime: btc_weight = w_trend_btc (default 1.0), sol_weight = 1 - btc_weight
   - In RANGE regime: btc_weight = w_range_btc (default 0.0), sol_weight = 1 - btc_weight
6) Save detailed CSV outputs for analysis.

Outputs
- results/portfolio_equity_<name>.csv
- results/portfolio_monthly_<name>.csv
- results/portfolio_regime_<name>.csv
- results/portfolio_trades_filtered_<name>.csv  (trades that occurred in their "active" regime)
- results/portfolio_trades_all_<name>.csv       (all trades produced by both simulators)

Usage example
python3 code/strategies/envelope/backtest_portfolio_regime_switch_cached.py \
  --btc-bot code/strategies/envelope/run_btc_trend_1h_v4.py \
  --sol-bot code/strategies/envelope/run_sol_bbrsi_1h_v3.py \
  --start "2025-02-28 00:00:00" \
  --end   "2026-02-28 00:00:00" \
  --initial 1000 \
  --name regime_v1 \
  --adx-period 14 --adx-on 27 --adx-off 23 \
  --vol-period 14 --vol-on 0.020 --vol-off 0.017 \
  --w-trend-btc 1.0 --w-range-btc 0.0
"""

from __future__ import annotations
import argparse, ast, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

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
    """
    Returns a dataframe with regime state per timestamp:
    - state: "TREND" or "RANGE"
    - adx, atr, atrp (atr/close)
    Uses hysteresis:
    - switch to TREND when condition >= on threshold
    - switch to RANGE when condition <= off threshold
    """
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

def _find_trade_time_col(trades: pd.DataFrame) -> Optional[str]:
    if trades is None or trades.empty:
        return None
    for c in ["entry_timestamp","timestamp","open_timestamp","entry_time","time"]:
        if c in trades.columns:
            return c
    return None

def filter_trades_by_regime(trades: pd.DataFrame, regime_df: pd.DataFrame, active_state: str) -> pd.DataFrame:
    if trades is None or trades.empty:
        return trades
    tc = _find_trade_time_col(trades)
    if tc is None:
        return trades  # can't filter safely
    t = trades.copy()
    # ensure int timestamps
    t[tc] = pd.to_datetime(t[tc], errors="coerce")
    if np.issubdtype(t[tc].dtype, np.datetime64):
        # convert to ms utc if datetime-like
        t["_ts"] = (pd.to_datetime(t[tc], utc=True).astype("int64") // 1_000_000).astype("int64")
    else:
        t["_ts"] = pd.to_numeric(t[tc], errors="coerce").astype("Int64")
        t["_ts"] = t["_ts"].astype("Int64").fillna(pd.NA)
    r = regime_df[["timestamp","state"]].copy().sort_values("timestamp")
    # merge-asof to assign latest regime state at or before trade time
    t2 = pd.merge_asof(
        t.sort_values("_ts"),
        r.rename(columns={"timestamp":"_ts"}),
        on="_ts",
        direction="backward"
    )
    t2 = t2[t2["state"] == active_state].drop(columns=["_ts"]).reset_index(drop=True)
    return t2


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
    ap.add_argument("--mode", choices=["adx_only","adx_and_vol"], default="adx_and_vol")
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

    # Run both strategies with their own capital (we'll combine using dynamic weights on returns)
    btc_init = args.initial
    sol_init = args.initial

    btc_trades, btc_eq = bt.simulate_trend(btc_df, btc_base, btc_init, args.maker_fee, args.taker_fee, args.btc_leverage)
    sol_trades, sol_eq = bt.simulate_bbrsi(sol_df, sol_base, sol_init, args.maker_fee, args.taker_fee, args.sol_leverage)

    # Prepare aligned returns
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

    # Portfolio returns and equity
    df["ret_port"] = df["w_btc"] * df["ret_btc"] + df["w_sol"] * df["ret_sol"]
    df["equity"] = args.initial * (1.0 + df["ret_port"]).cumprod()

    equity_df = df[["timestamp","equity","w_btc","w_sol","state","adx","atrp","equity_btc","equity_sol"]].copy()
    met = summarize_equity(equity_df["equity"])

    # Filter trades to those that happened in their "active" regime
    btc_active = filter_trades_by_regime(pd.DataFrame(btc_trades), regime, active_state="TREND")
    sol_active = filter_trades_by_regime(pd.DataFrame(sol_trades), regime, active_state="RANGE")

    all_trades = []
    if isinstance(btc_trades, pd.DataFrame):
        btc_all = btc_trades.copy()
    else:
        btc_all = pd.DataFrame(btc_trades)
    if not btc_all.empty:
        btc_all["strategy"] = "BTC_TREND"
    if isinstance(sol_trades, pd.DataFrame):
        sol_all = sol_trades.copy()
    else:
        sol_all = pd.DataFrame(sol_trades)
    if not sol_all.empty:
        sol_all["strategy"] = "SOL_BBRSI"
    if not btc_all.empty:
        all_trades.append(btc_all)
    if not sol_all.empty:
        all_trades.append(sol_all)
    trades_all_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    active_trades = []
    if btc_active is not None and not btc_active.empty:
        btc_active = btc_active.copy()
        btc_active["strategy"] = "BTC_TREND"
        active_trades.append(btc_active)
    if sol_active is not None and not sol_active.empty:
        sol_active = sol_active.copy()
        sol_active["strategy"] = "SOL_BBRSI"
        active_trades.append(sol_active)
    trades_active_df = pd.concat(active_trades, ignore_index=True) if active_trades else pd.DataFrame()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    name = args.name
    equity_path = outdir / f"portfolio_equity_{name}.csv"
    monthly_path = outdir / f"portfolio_monthly_{name}.csv"
    regime_path = outdir / f"portfolio_regime_{name}.csv"
    trades_all_path = outdir / f"portfolio_trades_all_{name}.csv"
    trades_active_path = outdir / f"portfolio_trades_filtered_{name}.csv"

    equity_df.to_csv(equity_path, index=False)
    monthly_summary(equity_df[["timestamp","equity"]], init_equity=args.initial).to_csv(monthly_path, index=False)
    regime.to_csv(regime_path, index=False)
    if not trades_all_df.empty:
        trades_all_df.to_csv(trades_all_path, index=False)
    if not trades_active_df.empty:
        trades_active_df.to_csv(trades_active_path, index=False)

    # Print summary
    print(f"\n=== Portfolio (Regime Switch): {name} ===")
    for k, v in met.items():
        print(f"{k}: {v}")

    m = monthly_summary(equity_df[["timestamp","equity"]], init_equity=args.initial)
    # trades per month (active)
    if not trades_active_df.empty:
        tc = _find_trade_time_col(trades_active_df)
        if tc is not None:
            t = trades_active_df.copy()
            # normalize timestamps to UTC month key
            if np.issubdtype(t[tc].dtype, np.datetime64):
                t["dt"] = pd.to_datetime(t[tc], utc=True)
            else:
                t["dt"] = pd.to_datetime(pd.to_numeric(t[tc], errors="coerce"), unit="ms", utc=True)
            t["month"] = t["dt"].dt.strftime("%Y-%m")
            counts = t.groupby(["month","strategy"]).size().unstack(fill_value=0)
            counts["trades_total"] = counts.sum(axis=1)
            m = m.merge(counts.reset_index(), on="month", how="left").fillna(0)
    print("\n=== Monthly returns (%) + active trades ===")
    for _, r in m.iterrows():
        tr = int(r.get("trades_total", 0))
        btc_m = int(r.get("BTC_TREND", 0))
        sol_m = int(r.get("SOL_BBRSI", 0))
        print(f"{r['month']}: {r['ret_pct']:.2f}% (equity_end={r['equity_end']:.2f}) | trades={tr} (BTC={btc_m}, SOL={sol_m})")

    print(f"\nSaved -> {equity_path}")
    print(f"Saved -> {monthly_path}")
    print(f"Saved -> {regime_path}")
    if not trades_all_df.empty:
        print(f"Saved -> {trades_all_path}")
    if not trades_active_df.empty:
        print(f"Saved -> {trades_active_path}")

if __name__ == "__main__":
    main()
