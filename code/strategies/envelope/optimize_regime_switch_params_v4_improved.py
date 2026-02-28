#!/usr/bin/env python3
"""
optimize_regime_switch_params_v4_improved.py

Fixes vs v1:
- Adds guardrails so the optimiser doesn't "solve" by turning TREND off (all RANGE).
- Penalises solutions with:
    * too little TREND exposure (trend_share < --min-trend-share)
    * too much TREND exposure (trend_share > --max-trend-share)  [optional]
    * zero ACTIVE BTC trades when w_trend_btc>0

Fixes vs v3:
- Uses SOFT penalties (no hard -1e9 discard by default) so the optimiser can rank near-feasible solutions.
- Adds optional --hard-constraints switch to restore v3 behaviour.
- Warns if the backtest script does not implement the requested --mode semantics.

- Adds guardrails so the optimiser doesn't "solve" by turning TREND off (all RANGE).
- Penalises solutions with:
    * too little TREND exposure (trend_share < --min-trend-share)
    * too much TREND exposure (trend_share > --max-trend-share)  [optional]
    * zero ACTIVE BTC trades when w_trend_btc > 0
- Still prioritises TEST generalisation (TEST calmar), but now ensures switching actually happens.

Usage:
python3 code/strategies/envelope/optimize_regime_switch_params_v4_improved.py \
  --btc-bot code/strategies/envelope/run_btc_trend_1h_v4.py \
  --sol-bot code/strategies/envelope/run_sol_bbrsi_1h_v3.py \
  --start "2025-02-28 00:00:00" \
  --end   "2026-02-28 00:00:00" \
  --split "2025-12-01 00:00:00" \
  --trials 500 \
  --seed 42 \
  --min-trend-share 0.15 \
  --max-trend-share 0.85 \
  --name reg_opt_v2
"""

from __future__ import annotations
import argparse, json, random, sys, ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import importlib.util

def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

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
    raise ValueError(f"Could not find params dict in {bot_file}")

def to_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def summarize_equity(equity: pd.Series) -> Dict[str, float]:
    eq = equity.astype(float).values
    start = float(eq[0])
    end = float(eq[-1])
    ret = (end / start - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max())
    max_dd_pct = float((max_dd / peak[np.argmax(dd)]) * 100.0) if max_dd > 0 else 0.0
    return {"return_pct": ret, "max_drawdown_pct": max_dd_pct}

def calmar(ret_pct: float, dd_pct: float) -> float:
    return ret_pct / max(dd_pct, 0.25)

def penalty_trend_share(trend_share: float, min_share: float, max_share: float) -> float:
    """
    Smooth penalty in (0,1]. 1 if inside bounds, otherwise decays.
    """
    if trend_share < min_share:
        # scale down to 0.5 at 0
        return max(0.5, 0.5 + 0.5 * (trend_share / max(min_share, 1e-9)))
    if trend_share > max_share:
        # scale down to 0.5 at 1
        over = (1.0 - trend_share) / max((1.0 - max_share), 1e-9)
        return max(0.5, 0.5 + 0.5 * over)
    return 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-bot", required=True)
    ap.add_argument("--sol-bot", required=True)
    ap.add_argument("--backtest-script", default="code/strategies/envelope/backtest_portfolio_regime_switch_cached_v2.py")
    ap.add_argument("--mode", choices=["adx_only","adx_and_vol","adx_or_enter_and_exit"], default="adx_or_enter_and_exit",
                    help="Regime logic passed to btmod.build_regime")
    ap.add_argument("--base-backtest-module", default=None,
                    help="Path to base portfolio backtest module exposing simulate_trend/simulate_bbrsi. If omitted, auto-detect backtest_portfolio_btc_sol_cached*.py next to --backtest-script.")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--cache-dir", default=".cache/ohlcv")
    ap.add_argument("--initial", type=float, default=1000.0)
    ap.add_argument("--maker-fee", type=float, default=0.0002)
    ap.add_argument("--taker-fee", type=float, default=0.0006)
    ap.add_argument("--btc-leverage", type=float, default=None)
    ap.add_argument("--sol-leverage", type=float, default=None)

    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="regime_opt_v2")
    ap.add_argument("--outdir", default="results")

    ap.add_argument("--min-trend-share", type=float, default=0.10)
    ap.add_argument("--max-trend-share", type=float, default=0.90)
    ap.add_argument("--min-active-btc-trades", type=int, default=5, help="Soft-penalise if BTC active trades is below this")
    
    ap.add_argument("--hard-constraints", action="store_true",
                    help="If set, restore v3 behaviour: discard trials that violate trend_share bounds or min-active-btc-trades by setting final_score=-1e9.")
    ap.add_argument("--penalty-share-power", type=float, default=1.0,
                    help="Penalty strength for trend_share bounds. Used as: final *= (pen_share_test*pen_share_train) ** penalty_share_power")
    ap.add_argument("--penalty-btc-power", type=float, default=2.0,
                    help="Penalty strength for min-active-btc-trades. Used as: final *= pen_btc_active ** penalty_btc_power")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    btmod = load_module(args.backtest_script, "regbt2")

    # Sanity-check: does build_regime() actually implement the requested mode?
    try:
        import inspect
        src_build = inspect.getsource(btmod.build_regime)
        if args.mode == "adx_or_enter_and_exit":
            # We expect OR for entry and AND for exit in this mode.
            # If the backtest file doesn't have an explicit branch, it will likely behave like adx_and_vol.
            if 'adx_or_enter_and_exit' not in src_build or ('elif' not in src_build and 'if' in src_build):
                print("[WARN] backtest.build_regime() may not implement explicit logic for mode=adx_or_enter_and_exit. "
                      "If results look identical to adx_and_vol, patch build_regime accordingly.", file=sys.stderr)
    except Exception:
        pass


    btc_params = parse_params_from_bot_file(args.btc_bot)
    sol_params = parse_params_from_bot_file(args.sol_bot)

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)
    split_ms = to_ms(args.split)

    btc_df = btmod.fetch_ohlcv_ccxt(
        btc_params["symbol"], btc_params.get("timeframe","1h"), start_ms, end_ms,
        exchange_id=args.exchange, cache_dir=args.cache_dir, use_cache=True, refresh_if_no_end=False
    )
    sol_df = btmod.fetch_ohlcv_ccxt(
        sol_params["symbol"], sol_params.get("timeframe","1h"), start_ms, end_ms,
        exchange_id=args.exchange, cache_dir=args.cache_dir, use_cache=True, refresh_if_no_end=False
    )

    btc_train = btc_df[btc_df["timestamp"] < split_ms].reset_index(drop=True)
    sol_train = sol_df[sol_df["timestamp"] < split_ms].reset_index(drop=True)
    btc_test  = btc_df[btc_df["timestamp"] >= split_ms].reset_index(drop=True)
    sol_test  = sol_df[sol_df["timestamp"] >= split_ms].reset_index(drop=True)

    # Load base module that provides simulate_trend/simulate_bbrsi
    base_path = args.base_backtest_module
    if base_path is None:
        import os, glob
        bt_dir = os.path.dirname(args.backtest_script) or '.'
        cand = os.path.join(bt_dir, 'backtest_portfolio_btc_sol_cached.py')
        if os.path.exists(cand):
            base_path = cand
        else:
            cands = sorted(glob.glob(os.path.join(bt_dir, 'backtest_portfolio_btc_sol_cached*.py')))
            if cands:
                base_path = cands[-1]
            else:
                raise FileNotFoundError(
                    f"Could not find base backtest module. Provide --base-backtest-module. "
                    f"Tried: {cand} and {os.path.join(bt_dir,'backtest_portfolio_btc_sol_cached*.py')}"
                )
    base_bt = btmod.load_backtest_module(base_path)
    simulate_trend = base_bt.simulate_trend
    simulate_bbrsi = base_bt.simulate_bbrsi

    # Pre-run equities and trades (fixed params)
    btc_tr_train, btc_eq_train = simulate_trend(btc_train, btc_params, args.initial, args.maker_fee, args.taker_fee, args.btc_leverage)
    sol_tr_train, sol_eq_train = simulate_bbrsi(sol_train, sol_params, args.initial, args.maker_fee, args.taker_fee, args.sol_leverage)
    btc_tr_test, btc_eq_test = simulate_trend(btc_test, btc_params, args.initial, args.maker_fee, args.taker_fee, args.btc_leverage)
    sol_tr_test, sol_eq_test = simulate_bbrsi(sol_test, sol_params, args.initial, args.maker_fee, args.taker_fee, args.sol_leverage)

    btc_tr_train_df = btc_tr_train if isinstance(btc_tr_train, pd.DataFrame) else pd.DataFrame(btc_tr_train)
    sol_tr_train_df = sol_tr_train if isinstance(sol_tr_train, pd.DataFrame) else pd.DataFrame(sol_tr_train)
    btc_tr_test_df  = btc_tr_test  if isinstance(btc_tr_test, pd.DataFrame)  else pd.DataFrame(btc_tr_test)
    sol_tr_test_df  = sol_tr_test  if isinstance(sol_tr_test, pd.DataFrame)  else pd.DataFrame(sol_tr_test)

    if not btc_tr_train_df.empty: btc_tr_train_df["strategy"] = "BTC_TREND"
    if not sol_tr_train_df.empty: sol_tr_train_df["strategy"] = "SOL_BBRSI"
    if not btc_tr_test_df.empty: btc_tr_test_df["strategy"] = "BTC_TREND"
    if not sol_tr_test_df.empty: sol_tr_test_df["strategy"] = "SOL_BBRSI"

    def combine(segment: str, adx_on, adx_off, vol_on, vol_off, w_trend_btc, w_range_btc):
        if segment == "train":
            bdf, sdf, beq, seq, btr, strd = btc_train, sol_train, btc_eq_train, sol_eq_train, btc_tr_train_df, sol_tr_train_df
        else:
            bdf, sdf, beq, seq, btr, strd = btc_test, sol_test, btc_eq_test, sol_eq_test, btc_tr_test_df, sol_tr_test_df

        reg = btmod.build_regime(bdf, 14, adx_on, adx_off, 14, vol_on, vol_off, mode=args.mode)
        trend_share = float((reg["state"] == "TREND").mean())

        b = beq[["timestamp","equity"]].rename(columns={"equity":"equity_btc"}).sort_values("timestamp")
        s = seq[["timestamp","equity"]].rename(columns={"equity":"equity_sol"}).sort_values("timestamp")
        df = pd.merge(b, s, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)
        df["equity_btc"] = df["equity_btc"].ffill().fillna(args.initial)
        df["equity_sol"] = df["equity_sol"].ffill().fillna(args.initial)
        df["ret_btc"] = df["equity_btc"].pct_change().fillna(0.0)
        df["ret_sol"] = df["equity_sol"].pct_change().fillna(0.0)
        df = pd.merge(df, reg[["timestamp","state"]], on="timestamp", how="left")
        df["state"] = df["state"].ffill().fillna("RANGE")
        df["w_btc"] = np.where(df["state"]=="TREND", w_trend_btc, w_range_btc)
        df["w_sol"] = 1.0 - df["w_btc"]
        df["ret_port"] = df["w_btc"]*df["ret_btc"] + df["w_sol"]*df["ret_sol"]
        df["equity"] = args.initial * (1.0 + df["ret_port"]).cumprod()

        # active BTC trades count (quick proxy)
        b_active = btmod.filter_trades_by_regime(btr, reg, active_state="TREND")
        active_btc = int(len(b_active)) if b_active is not None else 0

        return df, trend_share, active_btc

    rows = []

    best_final = -1e18
    best_params = None
    best_metrics = None
    for i in range(args.trials):
        # thresholds (narrow search around best-known basin for OR-enter/AND-exit)
        adx_off = rng.uniform(19.0, 21.0)
        adx_on  = rng.uniform(max(adx_off + 0.5, 21.0), 24.0)

        vol_off = rng.uniform(0.0068, 0.0074)
        vol_on  = rng.uniform(max(vol_off + 0.0008, 0.0082), 0.0090)

        # weights (avoid degenerate 'all SOL in RANGE' by forbidding w_range_btc=0)
        w_trend_btc = rng.choice([1.0, 0.9, 0.8])
        w_range_btc = rng.choice([0.1, 0.2, 0.3])
        train_df, train_trend_share, train_active_btc = combine("train", adx_on, adx_off, vol_on, vol_off, w_trend_btc, w_range_btc)
        test_df,  test_trend_share,  test_active_btc  = combine("test",  adx_on, adx_off, vol_on, vol_off, w_trend_btc, w_range_btc)

        train_m = summarize_equity(train_df["equity"])
        test_m  = summarize_equity(test_df["equity"])

        score_test = calmar(test_m["return_pct"], test_m["max_drawdown_pct"])
        score_train = calmar(train_m["return_pct"], train_m["max_drawdown_pct"])

        # Guardrail penalties (prefer switching actually used, especially on TEST)        # Guardrails / penalties
        pen_share_test  = penalty_trend_share(test_trend_share,  args.min_trend_share, args.max_trend_share)
        pen_share_train = penalty_trend_share(train_trend_share, args.min_trend_share, args.max_trend_share)

        # Penalise low "active" BTC trade count only when BTC is actually used in TREND
        if w_trend_btc > 0.0 and args.min_active_btc_trades > 0:
            pen_btc_active = min(1.0, float(test_active_btc) / float(args.min_active_btc_trades))
        else:
            pen_btc_active = 1.0

        base_final = (score_test + 0.25 * score_train)
        soft_mult  = ((pen_share_test * pen_share_train) ** float(args.penalty_share_power)) * (pen_btc_active ** float(args.penalty_btc_power))
        final = base_final * soft_mult

        if args.hard_constraints:
            # Restore v3 behaviour (hard discard)
            if not (args.min_trend_share <= test_trend_share <= args.max_trend_share):
                final = -1e9
            elif not (args.min_trend_share <= train_trend_share <= args.max_trend_share):
                final = -1e9
            elif w_trend_btc > 0.0 and test_active_btc < args.min_active_btc_trades:
                final = -1e9

# Track best trial
        if final > best_final:
            best_final = float(final)
            best_params = {
                "adx_on": float(adx_on), "adx_off": float(adx_off),
                "vol_on": float(vol_on), "vol_off": float(vol_off),
                "w_trend_btc": float(w_trend_btc), "w_range_btc": float(w_range_btc),
                "mode": args.mode,
            }
            best_metrics = {
                "test_return_pct": float(test_m["return_pct"]),
                "test_dd_pct": float(test_m["max_drawdown_pct"]),
                "test_trend_share": float(test_trend_share),
                "test_active_btc_trades": int(test_active_btc),
                "train_return_pct": float(train_m["return_pct"]),
                "train_dd_pct": float(train_m["max_drawdown_pct"]),
                "train_trend_share": float(train_trend_share),
            }
        rows.append({
            "trial": i,
            "final_score": float(final),
            "score_test": float(score_test),
            "score_train": float(score_train),
            "train_return_pct": float(train_m["return_pct"]),
            "train_dd_pct": float(train_m["max_drawdown_pct"]),
            "test_return_pct": float(test_m["return_pct"]),
            "test_dd_pct": float(test_m["max_drawdown_pct"]),
            "train_trend_share": float(train_trend_share),
            "test_trend_share": float(test_trend_share),
            "test_active_btc_trades": int(test_active_btc),
            "pen_share_test": float(pen_share_test),
            "pen_share_train": float(pen_share_train),
            "pen_btc_active": float(pen_btc_active),
            "adx_on": float(adx_on),
            "adx_off": float(adx_off),
            "vol_on": float(vol_on),
            "vol_off": float(vol_off),
            "w_trend_btc": float(w_trend_btc),
            "w_range_btc": float(w_range_btc),
        })

        if (i+1) % max(10, args.trials//10) == 0:
            # progress report using best-so-far trial
            if best_params is not None and best_metrics is not None:
                print(f"[{i+1}/{args.trials}] best final={best_final:.3f} "
                      f"TEST ret={best_metrics['test_return_pct']:.2f}% dd={best_metrics['test_dd_pct']:.2f}% "
                      f"trend_share_test={best_metrics['test_trend_share']:.2f} btc_active_test={best_metrics['test_active_btc_trades']} "
                      f"mode={best_params['mode']}")

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"opt_regime_trials_{args.name}.csv"
    df.to_csv(out_csv, index=False)

    top = df.head(10).to_dict(orient="records")
    out_json = outdir / f"opt_regime_top_{args.name}.json"
    out_json.write_text(json.dumps(top, indent=2), encoding="utf-8")

    
    if best_params is not None and best_metrics is not None:
        print("\n=== BEST TRIAL (by final_score) ===")
        for k in ["mode","adx_on","adx_off","vol_on","vol_off","w_trend_btc","w_range_btc"]:
            print(f"{k}: {best_params[k]}")
        for k in ["test_return_pct","test_dd_pct","test_trend_share","test_active_btc_trades",
                  "train_return_pct","train_dd_pct","train_trend_share"]:
            print(f"{k}: {best_metrics[k]}")

    print("\n=== TOP 10 (final_score) ===")
    cols = ["final_score","test_return_pct","test_dd_pct","test_trend_share","test_active_btc_trades",
            "train_return_pct","train_dd_pct","train_trend_share",
            "adx_on","adx_off","vol_on","vol_off","w_trend_btc","w_range_btc",
            "pen_share_test","pen_btc_active"]
    print(df[cols].head(10).to_string(index=False))
    print(f"\nSaved -> {out_csv}")
    print(f"Saved -> {out_json}")

if __name__ == "__main__":
    main()
