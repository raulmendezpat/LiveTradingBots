# SOL BB+RSI 1H v4 PROD (locked params, stable trackers)
# FIXES:
# - Adds robust OHLCV fetch fallback for BitgetFutures wrappers that don't expose fetch_ohlcv
# - Accepts either pandas.DataFrame or raw list-of-lists and normalizes to a DataFrame

import os
import sys
import json
from datetime import datetime

import ta
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utilities.bitget_futures import BitgetFutures

# ============================================================
# BB + RSI Mean Reversion (Single entry + re-entry after exit)
# Timeframe: 1H, Mode: both, Leverage: 2x, Sizing: % of balance
# ============================================================

params = {
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",  # "cross"
    "balance_fraction": 1,
    "leverage": 2,

    # Position sizing (keep your existing sizing style)
    "position_size_percentage": 20,  # % of balance used per trade (single entry)

    # Direction controls
    "use_longs": True,
    "use_shorts": True,

    # Bollinger Bands
    "bb_period": 20,
    "bb_std": 2.0,

    # RSI entry triggers
    "rsi_period": 14,
    "rsi_long_max": 36,   # allow longs when RSI <= this (oversold)
    "rsi_short_min": 64,  # allow shorts when RSI >= this (overbought)

    # Regime filter (avoid trending markets for mean reversion)
    "adx_period": 14,
    "adx_soft": 15,   # full size when ADX <= this
    "adx_hard": 22,   # no-trade when ADX >= this

    # ATR-based risk (recommended for 1H)
    "atr_period": 14,
    "stop_atr_mult": 1.8,  # stop distance = ATR * mult
    "tp_to_basis": True,   # TP at BB basis (middle band)

    # Entry order behavior (uses trigger-limit like your envelope bot)
    "trigger_price_delta": 0.004,  # closer for 1H
}

# Tracker to prevent duplicate order spam (simple, per-symbol)
TRACKER_FILE = os.path.join(
    os.path.dirname(__file__),
    "tracker_sol_bbrsi_1h_v4_prod.json",
)


def load_tracker():
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_tracker(state):
    try:
        with open(TRACKER_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"{datetime.now().strftime('%H:%M:%S')}: tracker save error: {e}")


def _to_ohlcv_df(data):
    """
    Normalizes raw OHLCV into a DataFrame with columns:
    timestamp, open, high, low, close, volume
    """
    if data is None:
        return None

    # Already a DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        # Ensure required columns exist
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"OHLCV DataFrame missing columns: {required - set(df.columns)}")
        # Ensure timestamp column exists (optional but used by tracker)
        if "timestamp" not in df.columns:
            if "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**6
            elif df.index.name in ("timestamp", "datetime"):
                # best-effort
                try:
                    df["timestamp"] = pd.to_datetime(df.index).astype("int64") // 10**6
                except Exception:
                    pass
        return df.reset_index(drop=True)

    # Raw list-of-lists (ccxt style): [ms, open, high, low, close, volume]
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple)):
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df

    raise ValueError(f"Unsupported OHLCV data type: {type(data)}")


def fetch_ohlcv_df(bitget: BitgetFutures, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    """
    BitgetFutures wrappers vary. Some don't expose fetch_ohlcv; we try common method names.
    Returns a normalized pandas DataFrame.
    """
    # 1) Preferred method name (your script originally used this)
    if hasattr(bitget, "fetch_ohlcv"):
        return _to_ohlcv_df(bitget.fetch_ohlcv(symbol, timeframe, limit=limit))

    # 2) Common alternatives
    candidates = [
        "fetch_ohlcv_dataframe",
        "fetch_ohlcv_df",
        "get_ohlcv",
        "get_ohlcv_df",
        "get_candles",
        "fetch_candles",
        "fetch_klines",
        "get_klines",
        "get_kline",
        "candles",
        "klines",
    ]
    for name in candidates:
        fn = getattr(bitget, name, None)
        if callable(fn):
            try:
                out = fn(symbol, timeframe, limit=limit)
            except TypeError:
                # some wrappers use (symbol, timeframe, n) positional without keyword
                out = fn(symbol, timeframe, limit)
            return _to_ohlcv_df(out)

    # 3) If BitgetFutures exposes the underlying ccxt exchange instance, try that
    for ex_attr in ("exchange", "client", "ccxt", "_exchange", "_client"):
        ex = getattr(bitget, ex_attr, None)
        if ex is not None and hasattr(ex, "fetch_ohlcv"):
            return _to_ohlcv_df(ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))

    raise AttributeError(
        "Could not find an OHLCV fetch method on BitgetFutures. "
        "Run an introspection: print([m for m in dir(BitgetFutures) if 'ohlc' in m.lower() or 'candle' in m.lower() or 'kline' in m.lower()])"
    )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Bollinger
    bb = ta.volatility.BollingerBands(df["close"], window=params["bb_period"], window_dev=params["bb_std"])
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_up"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=params["rsi_period"]).rsi()

    # ADX
    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=params["adx_period"])
    df["adx"] = adx_ind.adx()

    # ATR
    atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=params["atr_period"])
    df["atr"] = atr_ind.average_true_range()

    return df


def main():
    bitget = BitgetFutures()

    # 1) Fetch data (robust)
    data = fetch_ohlcv_df(bitget, params["symbol"], params["timeframe"], limit=300)
    if data is None or len(data) < 60:
        print(f"{datetime.now().strftime('%H:%M:%S')}: not enough data")
        return

    data = compute_indicators(data)

    # Use last CLOSED candle for signal to avoid repainting
    signal_idx = -2 if len(data) >= 2 else -1
    sig = data.iloc[signal_idx]

    # 2) Check open position
    position = bitget.get_open_position(params["symbol"], print_error=True)

    # 3) If a position exists: place TP to basis + SL (ATR) and exit (single entry logic)
    if position and position.get("contracts", 0) > 0:
        side = position["side"]  # "long" or "short"
        entry = float(position["info"]["openPriceAvg"])
        amount = position["contracts"] * position["contractSize"]

        basis = float(sig["bb_mid"])
        atr = float(sig["atr"])
        if atr != atr:
            print(f"{datetime.now().strftime('%H:%M:%S')}: ATR is NaN, skip exits this cycle")
            return

        if side == "long":
            close_side = "sell"
            stop_price = entry - (params["stop_atr_mult"] * atr)
        else:
            close_side = "buy"
            stop_price = entry + (params["stop_atr_mult"] * atr)

        # TP at basis (mean reversion)
        bitget.place_trigger_market_order(
            symbol=params["symbol"],
            side=close_side,
            amount=amount,
            trigger_price=basis,
            reduce=True,
            print_error=True,
        )

        # SL (ATR)
        bitget.place_trigger_market_order(
            symbol=params["symbol"],
            side=close_side,
            amount=amount,
            trigger_price=stop_price,
            reduce=True,
            print_error=True,
        )

        print(
            f"{datetime.now().strftime('%H:%M:%S')}: position {side} entry={entry:.4f} "
            f"TP(basis)={basis:.4f} SL(atr)={stop_price:.4f} amount={amount}"
        )
        return

    # 4) No position: decide whether to place a SINGLE entry order
    adx_v = float(sig["adx"])
    rsi_v = float(sig["rsi"])
    close_v = float(sig["close"])
    bb_low = float(sig["bb_low"])
    bb_up = float(sig["bb_up"])

    if any(x != x for x in [adx_v, rsi_v, bb_low, bb_up]):  # NaN guard
        print(f"{datetime.now().strftime('%H:%M:%S')}: indicators not ready")
        return

    if adx_v >= params["adx_hard"]:
        print(f"{datetime.now().strftime('%H:%M:%S')}: ADX={adx_v:.2f} >= {params['adx_hard']}, skip (trend regime)")
        return

    # Risk scaling: reduce size linearly when ADX is between soft and hard
    adx_soft = params.get("adx_soft", 15)
    adx_hard = params.get("adx_hard", 22)
    if adx_v <= adx_soft:
        adx_scale = 1.0
    else:
        adx_scale = max(0.0, min(1.0, 1.0 - (adx_v - adx_soft) / max(1e-9, (adx_hard - adx_soft))))

    # Balance -> notional -> qty
    bal = bitget.get_balance_usdt(print_error=True)
    if bal is None:
        print(f"{datetime.now().strftime('%H:%M:%S')}: could not read balance")
        return

    notional = (bal * (params["position_size_percentage"] / 100.0)) * params["leverage"] * adx_scale

    tracker = load_tracker()
    last_sig_ts = tracker.get("last_signal_ts")

    sig_ts = int(sig.get("timestamp", 0)) if "timestamp" in sig else None
    if sig_ts is not None and last_sig_ts == sig_ts:
        print(f"{datetime.now().strftime('%H:%M:%S')}: already processed this candle, skip")
        return

    placed = False

    # Long setup
    if params["use_longs"] and close_v <= bb_low and rsi_v <= params["rsi_long_max"]:
        entry_price = bb_low
        qty = notional / entry_price
        trigger_price = entry_price * (1 + params["trigger_price_delta"])

        bitget.place_trigger_limit_order(
            symbol=params["symbol"],
            side="buy",
            amount=qty,
            trigger_price=trigger_price,
            price=entry_price,
            print_error=True,
        )
        placed = True
        print(
            f"{datetime.now().strftime('%H:%M:%S')}: placed LONG entry at bb_low={entry_price:.4f} "
            f"trigger={trigger_price:.4f} RSI={rsi_v:.1f} ADX={adx_v:.1f} qty={qty:.6f}"
        )

    # Short setup
    elif params["use_shorts"] and close_v >= bb_up and rsi_v >= params["rsi_short_min"]:
        entry_price = bb_up
        qty = notional / entry_price
        trigger_price = entry_price * (1 - params["trigger_price_delta"])

        bitget.place_trigger_limit_order(
            symbol=params["symbol"],
            side="sell",
            amount=qty,
            trigger_price=trigger_price,
            price=entry_price,
            print_error=True,
        )
        placed = True
        print(
            f"{datetime.now().strftime('%H:%M:%S')}: placed SHORT entry at bb_up={entry_price:.4f} "
            f"trigger={trigger_price:.4f} RSI={rsi_v:.1f} ADX={adx_v:.1f} qty={qty:.6f}"
        )

    if sig_ts is not None:
        tracker["last_signal_ts"] = sig_ts
    tracker["last_action"] = "placed" if placed else "none"
    save_tracker(tracker)


if __name__ == "__main__":
    main()
