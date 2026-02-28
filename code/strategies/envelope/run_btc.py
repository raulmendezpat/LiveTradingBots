import os
import sys
import json
import ta
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utilities.bitget_futures import BitgetFutures

# ============================================================
# ✅ PRODUCTION CONFIG (aligned with backtesting)
# ============================================================
params = {
    'symbol': 'BTC/USDT:USDT',
    'timeframe': '4h',                       # ✅ CHANGED (was 1h)
    'margin_mode': 'isolated',               # 'cross'
    'balance_fraction': 1,
    'leverage': 2,                           # ✅ as per your deploy
    'average_type': 'EMA',                   # ✅ CHANGED (was SMA)
    'average_period': 20,                    # ✅ CHANGED (was 6)
    'envelopes': [0.025, 0.045],             # ✅ CHANGED (was [0.07,0.11,0.14])
    'stop_loss_pct': 0.03,                   # ✅ CHANGED (was 0.35)
    'price_jump_pct': 0.3,
    'position_size_percentage': 20,          # ✅ CHANGED (was 40) AND NOW USED PROPERLY
    'use_longs': True,
    'use_shorts': True,

    # ✅ NEW — Trend filter aligned with backtesting
    'trend_filter': True,
    'trend_filter_period': 100,              # ✅ EMA100 filter
    'trend_filter_directional': True,   # ✅ use price vs EMA(trend_filter_period)
    'max_extension_pct': 0.01,          # ✅ block shorts if price > EMA20 by 1% (4h)
    'trend_buffer_pct': 0.002,         # ✅ 0.2% buffer around EMA100 to avoid whipsaws
}

key_path = 'LiveTradingBots/secret.json'
key_name = 'envelope'

tracker_file = f"LiveTradingBots/code/strategies/envelope/tracker_{params['symbol'].replace('/', '-').replace(':', '-')}.json"

# ✅ CHANGED — trigger delta tuned for 4h (slightly wider than 1h to avoid noise triggers)
# You can adjust 0.008–0.012 based on fills/false triggers.
trigger_price_delta = 0.01

# ============================================================
# AUTH
# ============================================================
print(f"\n{datetime.now().strftime('%H:%M:%S')}: >>> starting execution for {params['symbol']}")
with open(key_path, "r") as f:
    api_setup = json.load(f)[key_name]
bitget = BitgetFutures(api_setup)

# ============================================================
# TRACKER FILE
# ============================================================
if not os.path.exists(tracker_file):
    with open(tracker_file, 'w') as file:
        json.dump({"status": "ok_to_trade", "last_side": None, "stop_loss_ids": []}, file)

def read_tracker_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def update_tracker_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)

# ============================================================
# CANCEL OPEN ORDERS
# ============================================================
orders = bitget.fetch_open_orders(params['symbol'])
for order in orders:
    bitget.cancel_order(order['id'], params['symbol'])

trigger_orders = bitget.fetch_open_trigger_orders(params['symbol'])
long_orders_left = 0
short_orders_left = 0
for order in trigger_orders:
    if order['side'] == 'buy' and order['info'].get('tradeSide') == 'open':
        long_orders_left += 1
    elif order['side'] == 'sell' and order['info'].get('tradeSide') == 'open':
        short_orders_left += 1
    bitget.cancel_trigger_order(order['id'], params['symbol'])

print(f"{datetime.now().strftime('%H:%M:%S')}: orders cancelled, {long_orders_left} longs left, {short_orders_left} shorts left")

# ============================================================
# FETCH OHLCV + INDICATORS
# ============================================================
# ✅ CHANGED — fetch more bars to properly warm up EMA100
data = bitget.fetch_recent_ohlcv(params['symbol'], params['timeframe'], 500).iloc[:-1]

if 'DCM' == params['average_type']:
    ta_obj = ta.volatility.DonchianChannel(data['high'], data['low'], data['close'], window=params['average_period'])
    data['average'] = ta_obj.donchian_channel_mband()
elif 'SMA' == params['average_type']:
    data['average'] = ta.trend.sma_indicator(data['close'], window=params['average_period'])
elif 'EMA' == params['average_type']:
    data['average'] = ta.trend.ema_indicator(data['close'], window=params['average_period'])
elif 'WMA' == params['average_type']:
    data['average'] = ta.trend.wma_indicator(data['close'], window=params['average_period'])
else:
    raise ValueError(f"The average type {params['average_type']} is not supported")

# ✅ NEW — Trend EMA (EMA100) for production filter
if params.get("trend_filter", False):
    data["trend_ema"] = ta.trend.ema_indicator(
        data["close"],
        window=params.get("trend_filter_period", 100)
    )

for i, e in enumerate(params['envelopes']):
    data[f'band_high_{i + 1}'] = data['average'] / (1 - e)
    data[f'band_low_{i + 1}'] = data['average'] * (1 - e)

print(f"{datetime.now().strftime('%H:%M:%S')}: ohlcv data fetched")

# ✅ NEW — Signal index = last CLOSED candle (avoid using the still-forming candle)
signal_idx = -2 if len(data) >= 2 else -1

# ✅ NEW — Trend permission (DIRECTIONAL) based on price vs EMA(trend_filter_period)
allow_long_by_trend = True
allow_short_by_trend = True

if params.get("trend_filter", False):
    price = float(data["close"].iloc[signal_idx])
    avg = float(data["average"].iloc[signal_idx])
    trn = float(data["trend_ema"].iloc[signal_idx])

    # NaN guard
    if (price != price) or (avg != avg) or (trn != trn):
        allow_long_by_trend = False
        allow_short_by_trend = False
    else:
        # Directional regime filter
        if params.get("trend_filter_directional", True):
            buffer = float(params.get("trend_buffer_pct", 0.0))
            if price > trn * (1 + buffer):
                allow_long_by_trend = True
                allow_short_by_trend = False
            elif price < trn * (1 - buffer):
                allow_long_by_trend = False
                allow_short_by_trend = True
            else:
                # Neutral zone around EMA(trend) to avoid whipsaws
                allow_long_by_trend = False
                allow_short_by_trend = False
        else:
            # (legacy behavior) compare average vs trend_ema
            allow_long_by_trend = avg > trn
            allow_short_by_trend = avg < trn

        # Anti-momentum / extension filter (reduces shorting into strong pumps)
        # NOTE: For envelope mean-reversion, blocking LONGs by distance-to-EMA(avg) can suppress valid entries.
        max_ext = params.get("max_extension_pct", None)
        if max_ext is not None:
            if allow_short_by_trend and price > avg * (1 + max_ext):
                allow_short_by_trend = False

    print(
        f"{datetime.now().strftime('%H:%M:%S')}: trend_filter EMA{params['trend_filter_period']} "
        f"price={price:.2f} avg(EMA{params['average_period']})={avg:.2f} trend={trn:.2f} "
        f"buffer={params.get('trend_buffer_pct', 0.0)} max_ext={params.get('max_extension_pct', None)} "
        f"allow_long={allow_long_by_trend} allow_short={allow_short_by_trend}"
    )

# ============================================================
# CHECK IF STOP LOSS WAS TRIGGERED

# ============================================================
closed_orders = bitget.fetch_closed_trigger_orders(params['symbol'])
tracker_info = read_tracker_file(tracker_file)

# NOTE: some APIs return newest first; keep your original logic but it's safer to scan all:
closed_ids = {o['id'] for o in closed_orders} if closed_orders else set()

if tracker_info.get('stop_loss_ids') and any(sl_id in closed_ids for sl_id in tracker_info['stop_loss_ids']):
    # pick last_side conservatively from latest closed order that matches
    matched = None
    for o in reversed(closed_orders):
        if o['id'] in tracker_info['stop_loss_ids']:
            matched = o
            break

    update_tracker_file(tracker_file, {
        "last_side": matched['info'].get('posSide') if matched else tracker_info.get("last_side"),
        "status": "stop_loss_triggered",
        "stop_loss_ids": [],
    })
    print(f"{datetime.now().strftime('%H:%M:%S')}: /!\\ stop loss was triggered")

# ============================================================
# CHECK MULTIPLE OPEN POSITIONS
# ============================================================
positions = bitget.fetch_open_positions(params['symbol'])
if positions:
    sorted_positions = sorted(positions, key=lambda x: x['timestamp'], reverse=True)
    latest_position = sorted_positions[0]
    for pos in sorted_positions[1:]:
        bitget.flash_close_position(pos['symbol'], side=pos['side'])
        print(f"{datetime.now().strftime('%H:%M:%S')}: double position case, closing the {pos['side']}.")

# ============================================================
# CHECK IF A POSITION IS OPEN
# ============================================================
position = bitget.fetch_open_positions(params['symbol'])
open_position = True if len(position) > 0 else False
if open_position:
    position = position[0]
    print(
        f"{datetime.now().strftime('%H:%M:%S')}: {position['side']} position of "
        f"{round(position['contracts'] * position['contractSize'], 2)} ~ "
        f"{round(position['contracts'] * position['contractSize'] * position['markPrice'], 2)} USDT is running"
    )

# ============================================================
# CLOSE ALL (price jump)
# ============================================================
if 'price_jump_pct' in params and open_position:
    if position['side'] == 'long':
        if data['close'].iloc[-1] < float(position['info']['openPriceAvg']) * (1 - params['price_jump_pct']):
            bitget.flash_close_position(params['symbol'])
            update_tracker_file(tracker_file, {
                "last_side": "long",
                "status": "close_all_triggered",
                "stop_loss_ids": [],
            })
            print(f"{datetime.now().strftime('%H:%M:%S')}: /!\\ close all was triggered")

    elif position['side'] == 'short':
        if data['close'].iloc[-1] > float(position['info']['openPriceAvg']) * (1 + params['price_jump_pct']):
            bitget.flash_close_position(params['symbol'])
            update_tracker_file(tracker_file, {
                "last_side": "short",
                "status": "close_all_triggered",
                "stop_loss_ids": [],
            })
            print(f"{datetime.now().strftime('%H:%M:%S')}: /!\\ close all was triggered")

# ============================================================
# OK TO TRADE CHECK
# ============================================================
tracker_info = read_tracker_file(tracker_file)
print(f"{datetime.now().strftime('%H:%M:%S')}: okay to trade check, status was {tracker_info['status']}")

last_price = data['close'].iloc[-1]
resume_price = data['average'].iloc[-1]

if tracker_info['status'] != "ok_to_trade":
    if ('long' == tracker_info['last_side'] and last_price >= resume_price) or (
            'short' == tracker_info['last_side'] and last_price <= resume_price):
        update_tracker_file(tracker_file, {"status": "ok_to_trade", "last_side": tracker_info['last_side']})
        print(f"{datetime.now().strftime('%H:%M:%S')}: status is now ok_to_trade")
    else:
        print(f"{datetime.now().strftime('%H:%M:%S')}: <<< status is still {tracker_info['status']}")
        sys.exit()

# ============================================================
# SET MARGIN MODE + LEVERAGE
# ============================================================
if not open_position:
    bitget.set_margin_mode(params['symbol'], margin_mode=params['margin_mode'])
    bitget.set_leverage(params['symbol'], margin_mode=params['margin_mode'], leverage=params['leverage'])

# ============================================================
# IF OPEN POSITION: REPLACE EXIT + SL
# ============================================================
if open_position:
    if position['side'] == 'long':
        close_side = 'sell'
        stop_loss_price = float(position['info']['openPriceAvg']) * (1 - params['stop_loss_pct'])
    elif position['side'] == 'short':
        close_side = 'buy'
        stop_loss_price = float(position['info']['openPriceAvg']) * (1 + params['stop_loss_pct'])

    amount = position['contracts'] * position['contractSize']

    # exit to average
    bitget.place_trigger_market_order(
        symbol=params['symbol'],
        side=close_side,
        amount=amount,
        trigger_price=data['average'].iloc[-1],
        reduce=True,
        print_error=True,
    )

    # stop-loss
    sl_order = bitget.place_trigger_market_order(
        symbol=params['symbol'],
        side=close_side,
        amount=amount,
        trigger_price=stop_loss_price,
        reduce=True,
        print_error=True,
    )

    info = {
        "status": "ok_to_trade",
        "last_side": position['side'],
        "stop_loss_price": stop_loss_price,
        "stop_loss_ids": [sl_order['id']] if sl_order else [],
    }

    print(
        f"{datetime.now().strftime('%H:%M:%S')}: placed close {position['side']} orders: "
        f"exit price {data['average'].iloc[-1]}, sl price {stop_loss_price}"
    )
else:
    info = {
        "status": "ok_to_trade",
        "last_side": tracker_info['last_side'],
        "stop_loss_ids": [],
    }

# ============================================================
# BALANCE / POSITION SIZING  (✅ FIXED)
# ============================================================
# ✅ CHANGED — Do NOT multiply wallet by leverage as "balance".
# Instead: wallet = USDT total * balance_fraction
wallet_usdt = params['balance_fraction'] * bitget.fetch_balance()['USDT']['total']

# ✅ NEW — This is the total NOTIONAL you want to deploy (wallet * leverage * %)
target_notional = wallet_usdt * params['leverage'] * (params['position_size_percentage'] / 100.0)

# Per envelope notional (split across envelopes)
per_envelope_notional = target_notional / len(params['envelopes'])

print(
    f"{datetime.now().strftime('%H:%M:%S')}: wallet_usdt={wallet_usdt:.2f} "
    f"target_notional={target_notional:.2f} per_env_notional={per_envelope_notional:.2f}"
)

# ============================================================
# PLACE ORDERS DEPENDING ON BANDS HIT + TREND FILTER
# ============================================================
if open_position:
    long_ok = True if 'long' == position['side'] else False
    short_ok = True if 'short' == position['side'] else False
    range_longs = range(len(params['envelopes']) - long_orders_left, len(params['envelopes']))
    range_shorts = range(len(params['envelopes']) - short_orders_left, len(params['envelopes']))
else:
    long_ok = True
    short_ok = True
    range_longs = range(len(params['envelopes']))
    range_shorts = range(len(params['envelopes']))

if not params['use_longs']:
    long_ok = False
if not params['use_shorts']:
    short_ok = False

# ✅ NEW — apply trend filter to permissions
if params.get("trend_filter", False):
    long_ok = long_ok and allow_long_by_trend
    short_ok = short_ok and allow_short_by_trend

# LONGS
if long_ok:
    for i in range_longs:
        entry_limit_price = float(data[f'band_low_{i + 1}'].iloc[signal_idx])
        entry_trigger_price = (1 + trigger_price_delta) * entry_limit_price

        amount = per_envelope_notional / entry_limit_price
        min_amount = bitget.fetch_min_amount_tradable(params['symbol'])

        if amount <= 0:
            amount = 2 * min_amount

        if amount >= min_amount:
            # entry
            bitget.place_trigger_limit_order(
                symbol=params['symbol'],
                side='buy',
                amount=amount,
                trigger_price=entry_trigger_price,
                price=entry_limit_price,
                print_error=True,
            )
            print(
                f"{datetime.now().strftime('%H:%M:%S')}: placed open long trigger limit order "
                f"env={i+1} amount={amount} trigger={entry_trigger_price} price={entry_limit_price}"
            )

            # exit to average
            bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side='sell',
                amount=amount,
                trigger_price=float(data['average'].iloc[signal_idx]),
                reduce=True,
                print_error=True,
            )

            # sl
            sl_price = entry_limit_price * (1 - params['stop_loss_pct'])
            sl_order = bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side='sell',
                amount=amount,
                trigger_price=sl_price,
                reduce=True,
                print_error=True,
            )
            if sl_order:
                info["stop_loss_ids"].append(sl_order['id'])

            print(
                f"{datetime.now().strftime('%H:%M:%S')}: placed exit+sl long orders "
                f"env={i+1} exit={float(data['average'].iloc[signal_idx])} sl={sl_price}"
            )
        else:
            print(
                f"{datetime.now().strftime('%H:%M:%S')}: /!\\ long orders not placed for env={i+1}, "
                f"amount {amount} < min {min_amount}"
            )

# SHORTS
if short_ok:
    for i in range_shorts:
        entry_limit_price = float(data[f'band_high_{i + 1}'].iloc[signal_idx])
        entry_trigger_price = (1 - trigger_price_delta) * entry_limit_price

        amount = per_envelope_notional / entry_limit_price
        min_amount = bitget.fetch_min_amount_tradable(params['symbol'])

        if amount <= 0:
            amount = 2 * min_amount

        if amount >= min_amount:
            # entry
            bitget.place_trigger_limit_order(
                symbol=params['symbol'],
                side='sell',
                amount=amount,
                trigger_price=entry_trigger_price,
                price=entry_limit_price,
                print_error=True,
            )
            print(
                f"{datetime.now().strftime('%H:%M:%S')}: placed open short trigger limit order "
                f"env={i+1} amount={amount} trigger={entry_trigger_price} price={entry_limit_price}"
            )

            # exit to average
            bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side='buy',
                amount=amount,
                trigger_price=float(data['average'].iloc[signal_idx]),
                reduce=True,
                print_error=True,
            )

            # sl
            sl_price = entry_limit_price * (1 + params['stop_loss_pct'])
            sl_order = bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side='buy',
                amount=amount,
                trigger_price=sl_price,
                reduce=True,
                print_error=True,
            )
            if sl_order:
                info["stop_loss_ids"].append(sl_order['id'])

            print(
                f"{datetime.now().strftime('%H:%M:%S')}: placed exit+sl short orders "
                f"env={i+1} exit={float(data['average'].iloc[signal_idx])} sl={sl_price}"
            )
        else:
            print(
                f"{datetime.now().strftime('%H:%M:%S')}: /!\\ short orders not placed for env={i+1}, "
                f"amount {amount} < min {min_amount}"
            )

update_tracker_file(tracker_file, info)
print(f"{datetime.now().strftime('%H:%M:%S')}: <<< all done")
