#!/usr/bin/env python3
# Regime Strategy v1.1 (1H, SOL) - based on v1, tuned for stability + controlled aggressiveness

params = {
    "name": "run_sol_regime_1h_v1_1",
    "exchange": "bitget",
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    "position_size_percentage": 14,
    "base_leverage": 2.0,
    "vol_lookback_hours": 168,
    "use_longs": True,
    "use_shorts": True,

    "ema_fast": 34,
    "ema_slow": 144,
    "ema_sep_trend": 0.0021,
    "ema_sep_chop": 0.0015,

    "donchian_period": 18,
    "breakout_confirm": 0.0008,
    "atr_period": 14,
    "adx_period": 14,
    "adx_trend_min": 18,
    "trail_atr_mult": 2.2,
    "trend_tp_atr_mult": 3.0,
    "max_hold_bars": 84,

    "bb_period": 20,
    "bb_std": 2.6,
    "rsi_period": 14,
    "rsi_long_max": 32,
    "rsi_short_min": 68,
    "mr_stop_atr_mult": 1.7,
    "mr_tp_mid": True,

    "min_atr_pct": 0.0009,
    "cooldown_bars_after_exit": 2,

    "dd_pause_pct": 0.14,
    "dd_pause_bars": 96,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 36,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
