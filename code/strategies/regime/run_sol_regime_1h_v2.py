#!/usr/bin/env python3
# Regime + Volatility Targeting Bot (1H, BOTH) - AGGRESSIVE MAX (v2)
# - Trend: Donchian breakout + ATR trailing stop (tighter, more turnover)
# - Mean reversion: BB + RSI (looser thresholds, more entries)
# - Vol target: higher target + higher max leverage

params = {
    "name": "run_sol_regime_1h_v2",

    "exchange": "bitget",
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # EXPOSURE (aggressive)
    "position_size_percentage": 45,     # was 25
    "use_longs": True,
    "use_shorts": True,

    # Leverage / vol targeting (aggressive)
    "base_leverage": 2.5,
    "target_vol_annual": 1.60,          # 160% annualized
    "vol_lookback_hours": 120,          # 5 days (faster response)
    "min_leverage": 1.0,
    "max_leverage": 10.0,               # was 5

    # Regime detection (less restrictive)
    "ema_fast": 40,
    "ema_slow": 180,
    "adx_period": 14,
    "adx_trend_min": 16,                # was 22
    "adx_chop_max": 26,                 # was 18

    # Trend breakout (more signals)
    "donchian_period": 10,              # was 20
    "atr_period": 14,
    "trail_atr_mult": 1.6,              # was 2.2
    "breakout_confirm": 0.0,
    "trend_leverage_boost": 1.60,       # was 1.20

    # Mean reversion (more entries)
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_long_max": 40,                 # was 32
    "rsi_short_min": 60,                # was 68
    "mr_stop_atr_mult": 1.10,           # was 1.4
    "mr_leverage_boost": 1.15,          # was 0.90

    # Fees
    "maker_fee": 0.0002,
    "taker_fee": 0.0006,

    # Safety
    "cooldown_bars_after_exit": 1,
}
