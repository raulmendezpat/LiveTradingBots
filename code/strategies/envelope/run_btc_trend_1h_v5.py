# BTC Trend 1H v5 (hysteresis ADX + pullback buffer + ATR trailing)
import os
import sys
import json
import ta
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.bitget_futures import BitgetFutures

# ============================================================
# BTC Trend Following 1H (Option B)
# EMA Pullback + ADX filter + ATR TP/SL
# - Single entry (1 position at a time) + re-entry after exit
# - Uses last CLOSED candle for signals
# ============================================================

params = {
    'symbol': 'BTC/USDT:USDT',
    'timeframe': '1h',
    'margin_mode': 'isolated',   # 'cross'
    'balance_fraction': 1,
    'leverage': 2,

    # Position sizing (% of balance, single entry)
    'position_size_percentage': 20,

    # Direction controls
    'use_longs': True,
    'use_shorts': True,

    # Trend + filter
    'ema_fast': 20,
    'ema_slow': 200,
    'adx_period': 14,
    'adx_enter_min': 27,        # enter/enable trend regime when ADX >= this
    'adx_exit_min': 22,         # keep trend regime until ADX < this (hysteresis)

    # ATR risk
    'atr_period': 14,
    'stop_atr_mult': 1.4,
    'tp_atr_mult': 2.6,

    # Entry quality / whipsaw control
    'entry_buffer_atr_mult': 0.20,   # require pullback beyond EMA20 by this * ATR

    # Trailing stop (tighten SL once trade moves in your favor)
    'trail_activate_atr': 1.0,       # start trailing after move >= this * ATR
    'trail_stop_atr_mult': 1.2,      # trailing distance = this * ATR

    # Entry behavior
    'trigger_price_delta': 0.0015,  # 0.2% trigger distance for 1H
}

TRACKER_FILE = os.path.join(os.path.dirname(__file__), "tracker_trend_btc.json")


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


def compute_indicators(df):
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=params["ema_fast"]).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=params["ema_slow"]).ema_indicator()

    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=params["adx_period"])
    df["adx"] = adx_ind.adx()

    atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=params["atr_period"])
    df["atr"] = atr_ind.average_true_range()
    return df


def main():
    bitget = BitgetFutures()

    data = bitget.fetch_ohlcv(params["symbol"], params["timeframe"], limit=350)
    if data is None or len(data) < 260:
        print(f"{datetime.now().strftime('%H:%M:%S')}: not enough data")
        return

    data = compute_indicators(data)

    # last CLOSED candle
    signal_idx = -2 if len(data) >= 2 else -1
    sig = data.iloc[signal_idx]

    # 1) If position exists: maintain TP/SL (ATR from signal candle)
    position = bitget.get_open_position(params["symbol"], print_error=True)
if position and position.get("contracts", 0) > 0:
    side = position["side"]  # 'long' or 'short'
    entry = float(position["info"]["openPriceAvg"])
    amount = position["contracts"] * position["contractSize"]

    atr = float(sig["atr"])
    close_now = float(sig["close"])
    if atr != atr:
        print(f"{datetime.now().strftime('%H:%M:%S')}: ATR NaN, skip exits")
        return

    if side == "long":
        close_side = "sell"
        tp = entry + params["tp_atr_mult"] * atr
        sl = entry - params["stop_atr_mult"] * atr
        move_atr = (close_now - entry) / atr if atr > 0 else 0.0
        if move_atr >= params.get("trail_activate_atr", 1.0):
            trail_sl = close_now - params.get("trail_stop_atr_mult", 1.2) * atr
            sl = max(sl, trail_sl)
    else:
        close_side = "buy"
        tp = entry - params["tp_atr_mult"] * atr
        sl = entry + params["stop_atr_mult"] * atr
        move_atr = (entry - close_now) / atr if atr > 0 else 0.0
        if move_atr >= params.get("trail_activate_atr", 1.0):
            trail_sl = close_now + params.get("trail_stop_atr_mult", 1.2) * atr
            sl = min(sl, trail_sl)

    # persist last trailing SL (best-effort; does not cancel older triggers)
    tracker = load_tracker()
    tracker["trail_sl"] = float(sl)
    save_tracker(tracker)

    bitget.place_trigger_market_order(
        symbol=params["symbol"],
        side=close_side,
        amount=amount,
        trigger_price=tp,
        reduce=True,
        print_error=True,
    )

    bitget.place_trigger_market_order(
        symbol=params["symbol"],
        side=close_side,
        amount=amount,
        trigger_price=sl,
        reduce=True,
        print_error=True,
    )

    print(f"{datetime.now().strftime('%H:%M:%S')}: pos {side} entry={entry:.2f} TP={tp:.2f} SL={sl:.2f} amount={amount}")
    return

    # 2) No position: decide entry (single entry + re-entry after exit)
    close_v = float(sig["close"])
    ema_fast = float(sig["ema_fast"])
    ema_slow = float(sig["ema_slow"])
    adx_v = float(sig["adx"])
    atr_v = float(sig["atr"])

    if any(x != x for x in [close_v, ema_fast, ema_slow, adx_v, atr_v]):
        print(f"{datetime.now().strftime('%H:%M:%S')}: indicators not ready")
        return

tracker = load_tracker()
tracker.setdefault("trend_regime", False)

in_regime = bool(tracker.get("trend_regime", False))
# ADX hysteresis: reduce on/off flicker
if not in_regime and adx_v < params["adx_enter_min"]:
    print(f"{datetime.now().strftime('%H:%M:%S')}: ADX={adx_v:.1f} < {params['adx_enter_min']} (enter), skip")
    return
if in_regime and adx_v < params["adx_exit_min"]:
    tracker["trend_regime"] = False
    save_tracker(tracker)
    print(f"{datetime.now().strftime('%H:%M:%S')}: ADX={adx_v:.1f} < {params['adx_exit_min']} (exit), disable trend regime")
    return
if (not in_regime) and adx_v >= params["adx_enter_min"]:
    tracker["trend_regime"] = True
    save_tracker(tracker)


    bal = bitget.get_balance_usdt(print_error=True)
    if bal is None:
        print(f"{datetime.now().strftime('%H:%M:%S')}: could not read balance")
        return

    notional = (bal * (params["position_size_percentage"] / 100.0)) * params["leverage"]

    tracker = load_tracker()
    last_sig_ts = tracker.get("last_signal_ts")
    sig_ts = int(sig.get("timestamp", 0)) if "timestamp" in sig else None
    if sig_ts is not None and last_sig_ts == sig_ts:
        print(f"{datetime.now().strftime('%H:%M:%S')}: already processed candle, skip")
        return

    placed = False

    buf = params.get("entry_buffer_atr_mult", 0.2) * atr_v

    # Long setup: trend up + pullback to EMA20 or below
    if params["use_longs"] and ema_fast > ema_slow and close_v <= (ema_fast - buf):
        entry_price = ema_fast  # pullback limit at EMA20
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
        print(f"{datetime.now().strftime('%H:%M:%S')}: placed LONG @EMA{params['ema_fast']}={entry_price:.2f} trigger={trigger_price:.2f} ADX={adx_v:.1f} qty={qty:.6f}")

    # Short setup: trend down + pullback to EMA20 or above
    elif params["use_shorts"] and ema_fast < ema_slow and close_v >= (ema_fast + buf):
        entry_price = ema_fast
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
        print(f"{datetime.now().strftime('%H:%M:%S')}: placed SHORT @EMA{params['ema_fast']}={entry_price:.2f} trigger={trigger_price:.2f} ADX={adx_v:.1f} qty={qty:.6f}")

    if sig_ts is not None:
        tracker["last_signal_ts"] = sig_ts
    tracker["last_action"] = "placed" if placed else "none"
    save_tracker(tracker)


if __name__ == "__main__":
    main()