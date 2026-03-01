#!/usr/bin/env python3
"""
stress_test_shock.py (v4)

Key fixes vs v3:
- Robust parsing of timestamp columns that are numeric Unix epochs (seconds/ms/us/ns).
- Better error messages when a forced --date-col doesn't exist.
- Uses resample("ME") (month-end) for Pandas versions where "M" is deprecated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


def equity_from_monthly(returns_pct: np.ndarray, initial: float) -> np.ndarray:
    eq = np.empty(len(returns_pct) + 1, dtype=float)
    eq[0] = initial
    eq[1:] = initial * np.cumprod(1.0 + returns_pct / 100.0)
    return eq


def max_drawdown(equity: np.ndarray) -> Tuple[float, float]:
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    mdd = float(dd.max()) if len(dd) else 0.0
    peak_at = float(peak[int(dd.argmax())]) if len(dd) else (float(peak[-1]) if len(peak) else 0.0)
    mdd_pct = (mdd / peak_at * 100.0) if peak_at > 0 else 0.0
    return mdd, mdd_pct


def summarize(returns_pct: np.ndarray, initial: float) -> dict:
    eq = equity_from_monthly(returns_pct, initial)
    end = float(eq[-1])
    ret = (end / initial - 1.0) * 100.0
    mdd, mdd_pct = max_drawdown(eq)
    return {
        "initial_equity": float(initial),
        "end_equity": end,
        "return_pct": ret,
        "max_dd": mdd,
        "max_dd_pct": mdd_pct,
        "months": int(len(returns_pct)),
    }


_DATE_CANDIDATES = ["datetime", "timestamp", "time", "date", "dt"]
_EQUITY_CANDIDATES = ["equity", "portfolio_equity", "equity_end", "value", "balance", "nav"]


def _pick_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_to_orig = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]
    for c in cols:
        cl = c.lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    return None


def _infer_epoch_unit(x: float) -> str:
    # Rough magnitude-based heuristic
    ax = abs(float(x))
    if ax < 1e11:
        return "s"   # seconds (~1.7e9 today)
    if ax < 1e14:
        return "ms"  # milliseconds (~1.7e12 today)
    if ax < 1e17:
        return "us"  # microseconds
    return "ns"      # nanoseconds


def _to_datetime_series(s: pd.Series) -> pd.Series:
    # Try numeric epoch first if it looks numeric
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:  # mostly numeric
        # infer unit from median magnitude
        med = float(s_num.dropna().median())
        unit = _infer_epoch_unit(med)
        dt = pd.to_datetime(s_num, errors="coerce", unit=unit, utc=True)
        return dt.dt.tz_convert(None)

    # Otherwise parse as strings/datetimes
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def monthly_from_equity_csv(equity_csv: str, date_col: Optional[str], equity_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(equity_csv)

    cols = df.columns.tolist()
    if date_col is None:
        date_col = _pick_column(cols, _DATE_CANDIDATES)
    if equity_col is None:
        equity_col = _pick_column(cols, _EQUITY_CANDIDATES)

    if date_col is None:
        raise SystemExit(
            f"equity CSV must contain a datetime-like column. Tried: {_DATE_CANDIDATES}. Got: {cols}"
        )
    if equity_col is None:
        raise SystemExit(
            f"equity CSV must contain an equity/value column. Tried: {_EQUITY_CANDIDATES}. Got: {cols}"
        )

    if date_col not in df.columns:
        raise SystemExit(f"--date-col '{date_col}' not found. Available columns: {cols}")
    if equity_col not in df.columns:
        raise SystemExit(f"--equity-col '{equity_col}' not found. Available columns: {cols}")

    df[date_col] = _to_datetime_series(df[date_col])
    df[equity_col] = pd.to_numeric(df[equity_col], errors="coerce")

    df = df.dropna(subset=[date_col, equity_col]).copy()
    df = df.sort_values(date_col)

    if df.empty:
        raise SystemExit("After parsing, equity CSV has no valid rows (datetime/equity). Check columns or formats.")

    s = df.set_index(date_col)[equity_col].astype(float)

    # Month-end resample (Pandas new versions use 'ME')
    month_end_eq = s.resample("ME").last().dropna()

    if len(month_end_eq) < 2:
        # Helpful diagnostics: show date range and some sample parsed dates
        dr0, dr1 = s.index.min(), s.index.max()
        sample = s.index[:5].tolist()
        raise SystemExit(
            f"Not enough month-end points after resample. Got {len(month_end_eq)}. "
            f"Parsed date range: {dr0} -> {dr1}. Sample parsed dates: {sample}. "
            f"Detected date_col='{date_col}', equity_col='{equity_col}'."
        )

    ret = month_end_eq.pct_change().dropna() * 100.0
    out = pd.DataFrame({"month": month_end_eq.index.strftime("%Y-%m"), "equity_end": month_end_eq.values})
    out = out.iloc[1:].copy()
    out["return_pct"] = ret.values
    return out.reset_index(drop=True)


def load_monthly_series(monthly_csv: Optional[str], equity_csv: Optional[str], date_col: Optional[str], equity_col: Optional[str]) -> pd.DataFrame:
    if monthly_csv:
        mdf = pd.read_csv(monthly_csv)
        if "return_pct" not in mdf.columns:
            raise SystemExit("monthly CSV must contain 'return_pct'")
        if "month" not in mdf.columns:
            mdf["month"] = [f"m{i+1}" for i in range(len(mdf))]
        mdf["return_pct"] = pd.to_numeric(mdf["return_pct"], errors="coerce")
        mdf = mdf.dropna(subset=["return_pct"]).reset_index(drop=True)
        if len(mdf) < 2:
            raise SystemExit("monthly CSV needs at least 2 rows of return_pct for meaningful stress testing.")
        return mdf

    if equity_csv:
        return monthly_from_equity_csv(equity_csv, date_col=date_col, equity_col=equity_col)

    raise SystemExit("Provide either --monthly-csv or --equity-csv")


def inject_worst(returns_pct: np.ndarray, shock_pct: float, initial: float) -> Tuple[np.ndarray, int, dict]:
    best_idx = 0
    worst_mdd_pct = -1.0
    best_series = None
    best_sum = None
    for idx in range(len(returns_pct) + 1):
        cand = np.insert(returns_pct, idx, shock_pct)
        s = summarize(cand, initial)
        if s["max_dd_pct"] > worst_mdd_pct:
            worst_mdd_pct = s["max_dd_pct"]
            best_idx = idx
            best_series = cand
            best_sum = s
    return best_series, best_idx, best_sum


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--monthly-csv", help="Monthly CSV with return_pct column (and optional month/equity_end)")
    grp.add_argument("--equity-csv", help="Equity curve CSV with datetime-like and equity/value columns")

    ap.add_argument("--date-col", default=None, help="Override: datetime column in --equity-csv")
    ap.add_argument("--equity-col", default=None, help="Override: equity/value column in --equity-csv")

    ap.add_argument("--initial", type=float, default=None, help="Initial equity. If omitted and --equity-csv is used, infer from first month.")
    ap.add_argument("--shock", type=float, default=-15.0, help="Shock month return in percent (e.g., -15)")
    ap.add_argument("--mode", choices=["append", "worst", "random"], default="worst")
    ap.add_argument("--sims", type=int, default=5000, help="Only for random mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None, help="Optional output CSV for stressed monthly series")
    ap.add_argument("--name", default=None, help="Optional label (only for printing)")
    args = ap.parse_args()

    mdf = load_monthly_series(args.monthly_csv, args.equity_csv, args.date_col, args.equity_col)
    base_r = mdf["return_pct"].to_numpy(dtype=float)

    if args.initial is not None:
        initial = float(args.initial)
    elif args.equity_csv and "equity_end" in mdf.columns and len(mdf["equity_end"]) > 0:
        first_end = float(mdf["equity_end"].iloc[0])
        first_ret = float(mdf["return_pct"].iloc[0])
        denom = (1.0 + first_ret / 100.0)
        initial = first_end / denom if denom != 0 else first_end
    else:
        initial = 1000.0

    label = args.name or Path(args.monthly_csv or args.equity_csv).stem

    base_sum = summarize(base_r, initial)
    print(f"\n=== Baseline: {label} ===")
    for k, v in base_sum.items():
        print(f"{k}: {v}")

    months = mdf["month"].astype(str).tolist() if "month" in mdf.columns else [f"m{i+1}" for i in range(len(base_r))]

    if args.mode == "append":
        stressed = np.append(base_r, args.shock)
        insert_idx = len(base_r)
        ssum = summarize(stressed, initial)
        print(f"\n=== Stressed (append one shock month {args.shock:.2f}%) ===")
        for k, v in ssum.items():
            print(f"{k}: {v}")
    elif args.mode == "worst":
        stressed, insert_idx, ssum = inject_worst(base_r, args.shock, initial)
        print(f"\n=== Stressed (worst-case timing, one shock month {args.shock:.2f}%) ===")
        print(f"insert_index: {insert_idx} (0 means before first month, {len(base_r)} means after last month)")
        for k, v in ssum.items():
            print(f"{k}: {v}")
    else:
        rng = np.random.default_rng(args.seed)
        n = len(base_r)
        out = np.empty((args.sims, 3), dtype=float)
        for i in range(args.sims):
            idx = int(rng.integers(0, n + 1))
            stressed_i = np.insert(base_r, idx, args.shock)
            s = summarize(stressed_i, initial)
            out[i] = (s["return_pct"], s["max_dd_pct"], s["end_equity"])
        df = pd.DataFrame(out, columns=["return_pct", "max_dd_pct", "end_equity"])
        print(f"\n=== Random placement of ONE shock month {args.shock:.2f}% (sims={args.sims}) ===")
        print(f"return_pct: mean={df['return_pct'].mean():.2f}, p5={df['return_pct'].quantile(0.05):.2f}, p50={df['return_pct'].quantile(0.50):.2f}, p95={df['return_pct'].quantile(0.95):.2f}")
        print(f"max_dd_pct: mean={df['max_dd_pct'].mean():.2f}, p95={df['max_dd_pct'].quantile(0.95):.2f}")
        return

    if args.save:
        months2 = months.copy()
        months2.insert(insert_idx, "SHOCK")
        out_df = pd.DataFrame({"month": months2, "return_pct": stressed})
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.save, index=False)
        print(f"\nSaved stressed monthly series -> {args.save}")


if __name__ == "__main__":
    main()
