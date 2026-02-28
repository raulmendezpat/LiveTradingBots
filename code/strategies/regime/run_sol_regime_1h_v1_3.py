#!/usr/bin/env python3
# Regime Strategy v1.3 (1H, SOL) - aligned to BTC v1.3
# SOL tends to move more; keep BB a bit wider than BTC to reduce churn.

params = {
    "name": "run_sol_regime_1h_v1_3",

    "exchange": "bitget",
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    "position_size_percentage": 20,
    "use_longs": True,
    "use_shorts": True,

    "base_leverage": 2.0,
    "target_vol_annual": 1.20,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 5.0,

    "ema_fast": 34,
    "ema_slow": 144,
    "adx_period": 14,
    "adx_trend_min": 18,
    "adx_chop_max": 24,

    "donchian_period": 16,
    "atr_period": 14,
    "trail_atr_mult": 2.2,
    "breakout_confirm": 0.0,
    "trend_leverage_boost": 1.15,

    "bb_period": 20,
    "bb_std": 2.05,
    "rsi_period": 14,
    "rsi_long_max": 36,
    "rsi_short_min": 64,
    "mr_stop_atr_mult": 1.45,
    "mr_leverage_boost": 0.95,

    "cooldown_bars_after_exit": 1,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
