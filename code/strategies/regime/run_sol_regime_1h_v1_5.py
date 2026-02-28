# -*- coding: utf-8 -*-
"""
Regime strategy – SOL 1h – v1.5

What changed vs v1.4 (high impact, low complexity):
1) Better risk/reward (SOL v1.4 had avg loss > avg win) by tightening SL and extending TP.
2) Gap / jump protection to avoid the rare but very large losses (some trades were labeled 'tp' but ended highly negative).
3) Cooldown after stop-loss to reduce chop-revenge sequences.
4) Volatility floor (min ATR%) to avoid trading when moves are too small relative to fees/slippage.

IMPORTANT:
- This file is intentionally "params-only" so it plugs into your existing backtest_regime.py loader
  in the same way your v1.4 run file does.
- If your Regime bot already supports these keys, you’re done.
- If not, implement support for the keys under `# REQUIRED ENGINE SUPPORT` below.
"""

# If your framework expects `BOT_NAME` or similar, keep it consistent:
BOT_NAME = "run_sol_regime_1h_v1_5"

params = {
    # --- Market ---
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",
    "leverage": 2,

    # --- Capital / sizing ---
    # Use a conservative fraction because SOL v1.4 drawdown was ~17.7%
    "balance_fraction": 1.0,
    "position_size_percentage": 20,     # was likely higher; reduce to stabilize

    # --- Regime core (leave your existing regime logic intact) ---
    # These are typical; keep if your bot already uses them
    "regime_lookback": 200,             # candles for regime estimation (trend vs range)
    "trend_strength_period": 14,        # ADX or similar
    "trend_strength_min": 18,           # higher = trade trend mode only when real trend exists

    # --- Risk/Reward (main fix) ---
    # v1.4 SOL losses averaged around -1.6% while wins ~ +1.35% => negative RR.
    # v1.5: tighten SL and push TP further so average RR can flip positive.
    "stop_loss_pct": 0.012,             # ~1.2% stop
    "take_profit_pct": 0.020,           # ~2.0% target

    # Optional: trailing stop can help trend-mode exits if your engine supports it
    "use_trailing_stop": True,
    "trailing_stop_pct": 0.010,         # trail at ~1.0%

    # --- Trade quality filters ---
    # REQUIRED ENGINE SUPPORT: skip entries when the market "jumps" beyond normal execution assumptions
    "enable_gap_filter": True,
    "max_entry_gap_pct": 0.020,         # if |open - prev_close|/prev_close > 2% => skip

    # REQUIRED ENGINE SUPPORT: avoid micro-volatility periods (fees kill small edges)
    "enable_min_atr_filter": True,
    "atr_period": 14,
    "min_atr_pct": 0.006,               # ATR/price must be >= 0.6%

    # REQUIRED ENGINE SUPPORT: cooldown after SL
    "enable_cooldown_after_sl": True,
    "cooldown_bars_after_sl": 6,         # 6 hours pause after SL

    # --- Side controls ---
    # In strong bull regimes SOL shorts can get punished; keep both but allow the regime logic to gate
    "use_longs": True,
    "use_shorts": True,

    # --- Execution assumptions (backtest realism) ---
    # If your engine supports fees/slippage, set them here (otherwise ignore)
    "taker_fee_pct": 0.0006,            # 0.06% (set to your real tier)
    "maker_fee_pct": 0.0002,            # 0.02%
    "slippage_pct": 0.0004,             # 0.04% per fill (conservative)

    # --- Backtester-required keys (added for backtest_regime.py compatibility) ---
    "ema_fast": 20,
    "ema_slow": 50,

    "atr_period": 14,
    "donchian_period": 20,

    "bb_period": 20,
    "bb_std": 2.0,

    "rsi_period": 14,
    "rsi_long_max": 35,
    "rsi_short_min": 65,

    "trail_atr_mult": 2.5,
    "mr_stop_atr_mult": 2.0,

    "adx_period": 14,
    "adx_trend_min": 18,
    "adx_chop_max": 16,

    "breakout_confirm": 0.0,
    "cooldown_bars_after_exit": 6,

    # Normalize fees (backtester reads maker_fee/taker_fee)
    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}

# REQUIRED ENGINE SUPPORT (if not already implemented in your regime bot):
#
# 1) gap filter:
#    - when you evaluate a new entry at candle open, compute:
#         gap = abs(open - prev_close) / prev_close
#      if gap > max_entry_gap_pct: do not open a new trade that candle.
#
# 2) min ATR filter:
#    - compute ATR(atr_period). If ATR/close < min_atr_pct: skip new entries.
#
# 3) cooldown after SL:
#    - when a trade closes with reason == "sl" (or pnl < 0 and close_type==stop),
#      set a counter = cooldown_bars_after_sl; block new entries while counter>0.
#
# 4) risk/reward:
#    - use stop_loss_pct for stop distance and take_profit_pct for target distance.
#
# 5) trailing stop (optional):
#    - if use_trailing_stop: update stop as price moves in favor.
