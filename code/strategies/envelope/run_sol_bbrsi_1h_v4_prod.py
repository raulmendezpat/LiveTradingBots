# SOL BB+RSI 1H v4 PROD (compatible with current BitgetFutures wrapper)
# Reference: production envelope bots run_sol.py / run_btc.py use:
# - bitget.fetch_recent_ohlcv(...)
# - bitget.fetch_open_positions(...)
# - bitget.fetch_balance()
#
# This script keeps the BB+RSI mean-reversion logic, but only uses those known-good methods.

import os
import sys
import json
from datetime import datetime

import ta
import pandas as pd

# Allow imports from /code
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
    "leverage": 3,

    # Position sizing (single entry)
    "position_size_percentage": 20,

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

    # ATR-based risk
    "atr_period": 14,
    "stop_atr_mult": 1.8,  # stop distance = ATR * mult
    "tp_to_basis": True,   # TP at BB basis (middle band)

    # Entry order behavior
    "trigger_price_delta": 0.004,  # closer for 1H
}

KEY_PATH = "LiveTradingBots/secret.json"
KEY_NAME = "envelope"  # same as run_sol.py / run_btc.py

TRACKER_FILE = os.path.join(
    os.path.dirname(__file__),
    "tracker_sol_bbrsi_1h_v4_prod.json",
)


def _now():
    return datetime.now().strftime("%H:%M:%S")


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
        print(f"{_now()}: tracker save error: {e}")


def get_bitget():
    # Matches production envelope bots (run_sol.py/run_btc.py)
    with open(KEY_PATH, "r") as f:
        api_setup = json.load(f)[KEY_NAME]
    return BitgetFutures(api_setup)


def fetch_ohlcv_df(bitget: BitgetFutures, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    """
    Uses the known-good method in your wrapper: fetch_recent_ohlcv.
    Assumes it returns a pandas.DataFrame with columns: open, high, low, close, volume (and maybe timestamp).
    """
    df = bitget.fetch_recent_ohlcv(symbol, timeframe, limit)
    if isinstance(df, pd.DataFrame) and len(df) > 1:
        # drop the current forming candle
        return df.iloc[:-1].reset_index(drop=True)
    return df


def get_open_position(bitget: BitgetFutures, symbol: str):
    """
    Wrapper uses fetch_open_positions(symbol) -> list.
    Return the first position dict if any, else None.
    """
    positions = bitget.fetch_open_positions(symbol)
    if not positions:
        return None
    return positions[0]


def wallet_usdt(bitget: BitgetFutures) -> float:
    bal = bitget.fetch_balance()
    usdt = bal.get("USDT", {})
    # prefer total; fallback to free
    return float(usdt.get("total", usdt.get("free", 0.0)))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(df["close"], window=params["bb_period"], window_dev=params["bb_std"])
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_up"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=params["rsi_period"]).rsi()

    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=params["adx_period"])
    df["adx"] = adx_ind.adx()

    atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=params["atr_period"])
    df["atr"] = atr_ind.average_true_range()
    return df


def main():
    print(f"\n{_now()}: >>> starting execution for {params['symbol']} (BB+RSI v4 PROD)")
    bitget = get_bitget()

    data = fetch_ohlcv_df(bitget, params["symbol"], params["timeframe"], limit=300)
    if data is None or len(data) < 60:
        print(f"{_now()}: not enough data")
        return

    data = compute_indicators(data)

    signal_idx = -1  # data already excludes forming candle
    sig = data.iloc[signal_idx]

    # 1) Open position?
    position = get_open_position(bitget, params["symbol"])

    # 2) If position exists: place TP at basis + SL (ATR) and exit
    if position and position.get("contracts", 0) > 0:
        side = position["side"]  # "long" or "short"
        entry = float(position["info"]["openPriceAvg"])
        amount = position["contracts"] * position["contractSize"]

        basis = float(sig["bb_mid"])
        atr = float(sig["atr"])
        if atr != atr:
            print(f"{_now()}: ATR is NaN, skip exits this cycle")
            return

        if side == "long":
            close_side = "sell"
            stop_price = entry - (params["stop_atr_mult"] * atr)
        else:
            close_side = "buy"
            stop_price = entry + (params["stop_atr_mult"] * atr)

        bitget.place_trigger_market_order(
            symbol=params["symbol"],
            side=close_side,
            amount=amount,
            trigger_price=basis,
            reduce=True,
            print_error=True,
        )

        bitget.place_trigger_market_order(
            symbol=params["symbol"],
            side=close_side,
            amount=amount,
            trigger_price=stop_price,
            reduce=True,
            print_error=True,
        )

        print(f"{_now()}: position {side} entry={entry:.4f} TP(basis)={basis:.4f} SL(atr)={stop_price:.4f} amount={amount}")
        return

    # 3) No position: decide entry
    adx_v = float(sig["adx"])
    rsi_v = float(sig["rsi"])
    close_v = float(sig["close"])
    bb_low = float(sig["bb_low"])
    bb_up = float(sig["bb_up"])

    if any(x != x for x in [adx_v, rsi_v, bb_low, bb_up]):
        print(f"{_now()}: indicators not ready")
        return

    if adx_v >= params["adx_hard"]:
        print(f"{_now()}: ADX={adx_v:.2f} >= {params['adx_hard']}, skip (trend regime)")
        return

    adx_soft = params.get("adx_soft", 15)
    adx_hard = params.get("adx_hard", 22)
    if adx_v <= adx_soft:
        adx_scale = 1.0
    else:
        adx_scale = max(0.0, min(1.0, 1.0 - (adx_v - adx_soft) / max(1e-9, (adx_hard - adx_soft))))

    w = wallet_usdt(bitget) * float(params.get("balance_fraction", 1))
    notional = w * params["leverage"] * (params["position_size_percentage"] / 100.0) * adx_scale

    tracker = load_tracker()
    last_sig_ts = tracker.get("last_signal_ts")

    sig_ts = int(sig.get("timestamp", 0)) if "timestamp" in sig else None
    if sig_ts is not None and last_sig_ts == sig_ts:
        print(f"{_now()}: already processed this candle, skip")
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
        print(f"{_now()}: placed LONG entry@bb_low={entry_price:.4f} trigger={trigger_price:.4f} RSI={rsi_v:.1f} ADX={adx_v:.1f} qty={qty:.6f}")

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
        print(f"{_now()}: placed SHORT entry@bb_up={entry_price:.4f} trigger={trigger_price:.4f} RSI={rsi_v:.1f} ADX={adx_v:.1f} qty={qty:.6f}")

    if sig_ts is not None:
        tracker["last_signal_ts"] = sig_ts
    tracker["last_action"] = "placed" if placed else "none"
    save_tracker(tracker)


if __name__ == "__main__":
    main()
