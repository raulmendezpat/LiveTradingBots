# run_sol_bbrsi_1h_v4_prod.py
# SOL BB+RSI 1H v4.1 PROD — anti-duplicate + pre-placed TP/SL bracket
#
# Wrapper methods used:
#   fetch_recent_ohlcv, fetch_open_positions, fetch_balance
#   fetch_open_trigger_orders, cancel_trigger_order
#   place_trigger_limit_order, place_trigger_market_order
#
# Key changes vs prior prod:
# 1) Anti-dup: candle timestamp from DataFrame index/column + tracker last_entry_candle_ts
# 2) Anti-dup: OS file lock prevents concurrent runs
# 3) Pre-place TP/SL reduce-only immediately after placing entry trigger (bracket)
#    - TP at BB mid; SL using expected entry ± ATR*mult
#    - TTL cleanup if entry not filled
# 4) If position exists: refresh TP/SL each run (cancel old reduce-only triggers then place updated exits)

import os
import sys
import json
import fcntl
from datetime import datetime

import ta
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utilities.bitget_futures import BitgetFutures

params = {
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",
    "balance_fraction": 1,
    "leverage": 2,

    "position_size_percentage": 20,
    "use_longs": True,
    "use_shorts": True,

    "bb_period": 20,
    "bb_std": 2.0,

    "rsi_period": 14,
    "rsi_long_max": 36,
    "rsi_short_min": 64,

    "adx_period": 14,
    "adx_soft": 15,
    "adx_hard": 22,

    "atr_period": 14,
    "stop_atr_mult": 1.8,

    "trigger_price_delta": 0.004,

    # bracket
    "preplace_bracket": True,
    "preplace_ttl_runs": 3,
}

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KEY_PATH = os.path.join(ROOT_DIR, "secret.json")
KEY_NAME = "envelope"

TRACKER_FILE = os.path.join(os.path.dirname(__file__), "tracker_sol_bbrsi_1h_v4_prod.json")
LOCK_FILE = "/tmp/sol_bbrsi_1h.lock"


def _now():
    return datetime.now().strftime("%H:%M:%S")


def acquire_lock():
    f = open(LOCK_FILE, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except BlockingIOError:
        return None


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
    with open(KEY_PATH, "r") as f:
        api_setup = json.load(f)[KEY_NAME]
    return BitgetFutures(api_setup)


def fetch_ohlcv_df(bitget: BitgetFutures, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    df = bitget.fetch_recent_ohlcv(symbol, timeframe, limit)
    if isinstance(df, pd.DataFrame) and len(df) > 1:
        return df.iloc[:-1].reset_index(drop=False)
    return df


def get_open_position(bitget: BitgetFutures, symbol: str):
    positions = bitget.fetch_open_positions(symbol)
    return positions[0] if positions else None


def wallet_usdt(bitget: BitgetFutures) -> float:
    bal = bitget.fetch_balance()
    usdt = bal.get("USDT", {})
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


def _extract_oid(o):
    if not isinstance(o, dict):
        return None
    return o.get("id") or o.get("orderId") or o.get("info", {}).get("orderId")


def _is_reduce_only(o):
    if not isinstance(o, dict):
        return False
    info = o.get("info", {}) if isinstance(o.get("info", {}), dict) else {}
    return bool(o.get("reduceOnly") or info.get("reduceOnly") or info.get("reduce") or o.get("reduce"))


def cancel_trigger_ids(bitget, symbol, ids):
    for oid in ids:
        if not oid:
            continue
        try:
            bitget.cancel_trigger_order(oid, symbol)
        except Exception as e:
            print(f"{_now()}: cancel_trigger_order failed for {oid}: {e}")


def cancel_all_reduce_only_triggers(bitget, symbol):
    try:
        orders = bitget.fetch_open_trigger_orders(symbol) or []
    except Exception:
        return
    for o in orders:
        if _is_reduce_only(o):
            oid = _extract_oid(o)
            if oid:
                try:
                    bitget.cancel_trigger_order(oid, symbol)
                except Exception:
                    pass


def candle_epoch_seconds(sig_row: pd.Series, df: pd.DataFrame):
    if "timestamp" in df.columns:
        try:
            return int(pd.Timestamp(sig_row["timestamp"]).timestamp())
        except Exception:
            pass
    try:
        return int(pd.Timestamp(sig_row.name).timestamp())
    except Exception:
        return None


def preplace_bracket(bitget, symbol, entry_side, amount, expected_entry, tp_price, atr, tracker):
    """
    entry_side: 'buy' or 'sell'
    TP uses tp_price (BB mid). SL uses expected_entry ± ATR*mult.
    """
    if atr != atr or atr <= 0:
        print(f"{_now()}: ATR invalid -> skip preplace bracket")
        return

    if entry_side == "buy":
        close_side = "sell"
        tp = tp_price
        sl = expected_entry - params["stop_atr_mult"] * atr
    else:
        close_side = "buy"
        tp = tp_price
        sl = expected_entry + params["stop_atr_mult"] * atr

    tp_res = bitget.place_trigger_market_order(
        symbol=symbol, side=close_side, amount=amount,
        trigger_price=tp, reduce=True, print_error=True
    )
    sl_res = bitget.place_trigger_market_order(
        symbol=symbol, side=close_side, amount=amount,
        trigger_price=sl, reduce=True, print_error=True
    )

    tracker["pre_bracket"] = {
        "runs_left": int(params.get("preplace_ttl_runs", 3)),
        "entry_side": entry_side,
        "expected_entry": float(expected_entry),
        "tp": float(tp),
        "sl": float(sl),
        "tp_id": _extract_oid(tp_res),
        "sl_id": _extract_oid(sl_res),
    }
    save_tracker(tracker)
    print(f"{_now()}: preplaced TP/SL for pending entry. TP={tp:.4f} SL={sl:.4f}")


def main():
    lock = acquire_lock()
    if not lock:
        print(f"{_now()}: another run is active -> skip")
        return

    print(f"\n{_now()}: >>> starting execution for {params['symbol']} (BB+RSI v4.1 PROD)")
    bitget = get_bitget()

    data = fetch_ohlcv_df(bitget, params["symbol"], params["timeframe"], limit=300)
    if data is None or len(data) < 60:
        print(f"{_now()}: not enough data")
        return

    data = compute_indicators(data)
    sig = data.iloc[-1]
    sig_ts = candle_epoch_seconds(sig, data)

    position = get_open_position(bitget, params["symbol"])

    # =========================
    # If position exists: exits
    # =========================
    if position and position.get("contracts", 0) > 0:
        tracker = load_tracker()

        pre = tracker.get("pre_bracket", {})
        cancel_trigger_ids(bitget, params["symbol"], [pre.get("tp_id"), pre.get("sl_id")])
        tracker.pop("pre_bracket", None)
        save_tracker(tracker)

        cancel_all_reduce_only_triggers(bitget, params["symbol"])

        side = position["side"]
        entry = float(position["info"]["openPriceAvg"])
        amount = position["contracts"] * position["contractSize"]

        basis = float(sig["bb_mid"])
        atr = float(sig["atr"])
        if atr != atr:
            print(f"{_now()}: ATR NaN, skip exits")
            return

        if side == "long":
            close_side = "sell"
            stop_price = entry - params["stop_atr_mult"] * atr
        else:
            close_side = "buy"
            stop_price = entry + params["stop_atr_mult"] * atr

        bitget.place_trigger_market_order(
            symbol=params["symbol"], side=close_side, amount=amount,
            trigger_price=basis, reduce=True, print_error=True
        )
        bitget.place_trigger_market_order(
            symbol=params["symbol"], side=close_side, amount=amount,
            trigger_price=stop_price, reduce=True, print_error=True
        )

        print(f"{_now()}: position {side} entry={entry:.4f} TP(basis)={basis:.4f} SL={stop_price:.4f} amount={amount}")
        return

    # ==============================
    # No position: anti-dup + bracket
    # ==============================
    tracker = load_tracker()

    pre = tracker.get("pre_bracket")
    if pre:
        pre["runs_left"] = int(pre.get("runs_left", 1)) - 1
        tracker["pre_bracket"] = pre
        save_tracker(tracker)
        if pre["runs_left"] <= 0:
            cancel_trigger_ids(bitget, params["symbol"], [pre.get("tp_id"), pre.get("sl_id")])
            tracker.pop("pre_bracket", None)
            save_tracker(tracker)
            print(f"{_now()}: canceled stale preplaced TP/SL (entry not filled)")

    if sig_ts is not None and tracker.get("last_entry_candle_ts") == sig_ts:
        print(f"{_now()}: already placed/attempted entry this candle -> skip")
        return

    adx_v = float(sig["adx"])
    rsi_v = float(sig["rsi"])
    close_v = float(sig["close"])
    bb_low = float(sig["bb_low"])
    bb_up = float(sig["bb_up"])
    basis = float(sig["bb_mid"])
    atr = float(sig["atr"])

    if any(x != x for x in [adx_v, rsi_v, bb_low, bb_up, basis, atr]):
        print(f"{_now()}: indicators not ready")
        return

    if adx_v >= params["adx_hard"]:
        print(f"{_now()}: ADX={adx_v:.2f} >= {params['adx_hard']}, skip (trend regime)")
        return

    # scale size by ADX between soft/hard
    adx_soft = params.get("adx_soft", 15)
    adx_hard = params.get("adx_hard", 22)
    if adx_v <= adx_soft:
        adx_scale = 1.0
    else:
        adx_scale = max(0.0, min(1.0, 1.0 - (adx_v - adx_soft) / max(1e-9, (adx_hard - adx_soft))))

    w = wallet_usdt(bitget) * float(params.get("balance_fraction", 1))
    notional = w * params["leverage"] * (params["position_size_percentage"] / 100.0) * adx_scale

    placed = False

    # Long setup
    if params["use_longs"] and close_v <= bb_low and rsi_v <= params["rsi_long_max"]:
        entry_price = bb_low
        qty = notional / entry_price
        trigger_price = entry_price * (1 + params["trigger_price_delta"])

        bitget.place_trigger_limit_order(
            symbol=params["symbol"], side="buy", amount=qty,
            trigger_price=trigger_price, price=entry_price, print_error=True
        )
        placed = True
        if params.get("preplace_bracket", True):
            preplace_bracket(bitget, params["symbol"], "buy", qty, entry_price, basis, atr, tracker)

        print(f"{_now()}: placed LONG entry@bb_low={entry_price:.4f} trigger={trigger_price:.4f} RSI={rsi_v:.1f} ADX={adx_v:.1f} qty={qty:.6f}")

    # Short setup
    elif params["use_shorts"] and close_v >= bb_up and rsi_v >= params["rsi_short_min"]:
        entry_price = bb_up
        qty = notional / entry_price
        trigger_price = entry_price * (1 - params["trigger_price_delta"])

        bitget.place_trigger_limit_order(
            symbol=params["symbol"], side="sell", amount=qty,
            trigger_price=trigger_price, price=entry_price, print_error=True
        )
        placed = True
        if params.get("preplace_bracket", True):
            preplace_bracket(bitget, params["symbol"], "sell", qty, entry_price, basis, atr, tracker)

        print(f"{_now()}: placed SHORT entry@bb_up={entry_price:.4f} trigger={trigger_price:.4f} RSI={rsi_v:.1f} ADX={adx_v:.1f} qty={qty:.6f}")

    if placed and sig_ts is not None:
        tracker["last_entry_candle_ts"] = sig_ts
        tracker["last_action"] = "placed_entry"
        save_tracker(tracker)


if __name__ == "__main__":
    main()
