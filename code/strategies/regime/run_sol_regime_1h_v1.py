#!/usr/bin/env python3
# Regime + Volatility Targeting Bot (1H, BOTH)
# - Trend regime: Donchian breakout with ATR trailing stop
# - Mean-reversion regime: BB(20,2) + RSI filter, exit to basis
# - Dynamic leverage: volatility targeting (target_vol_annual) with caps
# NOTE: This file only defines params for backtester / live runner.

params = {
    "name": "run_sol_regime_1h_v1",

    # Exchange / market
    "exchange": "bitget",
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Risk & sizing
    "position_size_percentage": 25,     # % of equity allocated per trade (notional BEFORE leverage)
    "use_longs": True,
    "use_shorts": True,

    # Base leverage (used as multiplier with vol-target leverage factor)
    "base_leverage": 2.0,

    # Volatility targeting (annualized)
    "target_vol_annual": 1.00,          # 100% annualized target vol
    "vol_lookback_hours": 168,          # 7 days on 1H
    "min_leverage": 1.0,
    "max_leverage": 5.0,

    # Regime detection
    "ema_fast": 50,
    "ema_slow": 200,
    "adx_period": 14,
    "adx_trend_min": 22,                # above => trend regime likely
    "adx_chop_max": 18,                 # below => allow mean reversion

    # --- Trend strategy (breakout) ---
    "donchian_period": 20,              # breakout window
    "atr_period": 14,
    "trail_atr_mult": 2.2,              # trailing stop distance
    "breakout_confirm": 0.0,            # extra buffer (e.g., 0.001 = 0.1%)
    "trend_leverage_boost": 1.20,       # boost leverage when trend regime strong

    # --- Mean reversion strategy ---
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_long_max": 32,
    "rsi_short_min": 68,
    "mr_stop_atr_mult": 1.4,            # hard stop for MR entries
    "mr_leverage_boost": 0.90,          # slightly lower than trend

    # Execution / fees assumptions (can override in backtest CLI)
    "maker_fee": 0.0002,
    "taker_fee": 0.0006,

    # Safety
    "cooldown_bars_after_exit": 1,       # avoid instant re-entry same bar
}
