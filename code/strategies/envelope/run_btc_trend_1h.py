
import os
import sys
import json
import ta
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.bitget_futures import BitgetFutures

# ============================================================
# BTC Trend Following 1H
# Pullback to EMA20 in direction of EMA200
# Single entry + re-entry
# ============================================================

params = {
    'symbol': 'BTC/USDT:USDT',
    'timeframe': '1h',
    'margin_mode': 'isolated',
    'balance_fraction': 1,
    'leverage': 2,

    'position_size_percentage': 20,

    'use_longs': True,
    'use_shorts': True,

    # Trend filters
    'ema_fast': 20,
    'ema_slow': 200,

    # ATR risk
    'atr_period': 14,
    'stop_atr_mult': 1.5,
    'tp_atr_mult': 2.0,

    'trigger_price_delta': 0.002
}

def compute_indicators(df):
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=params["ema_fast"]).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=params["ema_slow"]).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=params["atr_period"]
    ).average_true_range()
    return df

def main():
    bitget = BitgetFutures()
    data = bitget.fetch_ohlcv(params["symbol"], params["timeframe"], limit=300)
    if data is None or len(data) < 250:
        print("Not enough data")
        return

    data = compute_indicators(data)
    sig = data.iloc[-2]

    position = bitget.get_open_position(params["symbol"], print_error=True)

    if position and position.get("contracts", 0) > 0:
        side = position["side"]
        entry = float(position["info"]["openPriceAvg"])
        qty = position["contracts"] * position["contractSize"]
        atr = float(sig["atr"])

        if side == "long":
            tp = entry + params["tp_atr_mult"] * atr
            sl = entry - params["stop_atr_mult"] * atr
            close_side = "sell"
        else:
            tp = entry - params["tp_atr_mult"] * atr
            sl = entry + params["stop_atr_mult"] * atr
            close_side = "buy"

        bitget.place_trigger_market_order(
            symbol=params["symbol"],
            side=close_side,
            amount=qty,
            trigger_price=tp,
            reduce=True,
            print_error=True,
        )

        bitget.place_trigger_market_order(
            symbol=params["symbol"],
            side=close_side,
            amount=qty,
            trigger_price=sl,
            reduce=True,
            print_error=True,
        )
        return

    close = float(sig["close"])
    ema_fast = float(sig["ema_fast"])
    ema_slow = float(sig["ema_slow"])
    atr = float(sig["atr"])

    bal = bitget.get_balance_usdt(print_error=True)
    if bal is None:
        return

    notional = bal * (params["position_size_percentage"] / 100.0) * params["leverage"]

    # Long trend condition
    if params["use_longs"] and ema_fast > ema_slow and close <= ema_fast:
        qty = notional / close
        bitget.place_trigger_limit_order(
            symbol=params["symbol"],
            side="buy",
            amount=qty,
            trigger_price=close * (1 + params["trigger_price_delta"]),
            price=close,
            print_error=True,
        )

    # Short trend condition
    if params["use_shorts"] and ema_fast < ema_slow and close >= ema_fast:
        qty = notional / close
        bitget.place_trigger_limit_order(
            symbol=params["symbol"],
            side="sell",
            amount=qty,
            trigger_price=close * (1 - params["trigger_price_delta"]),
            price=close,
            print_error=True,
        )

if __name__ == "__main__":
    main()
