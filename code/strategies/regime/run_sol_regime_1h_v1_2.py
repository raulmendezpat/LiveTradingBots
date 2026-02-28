#!/usr/bin/env python3
# Regime Strategy v1.2 (1H, SOL) - aligned with BTC v1.2, slightly higher vol filter

params = {
    "name": "run_sol_regime_1h_v1_2",
    "exchange": "bitget",
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    "position_size_percentage": 14,
    "base_leverage": 2.0,
    "use_longs": True,
    "use_shorts": True,

    "vol_lookback_hours": 168,
    "target_vol_annual": 1.25,
    "min_leverage": 1.0,
    "max_leverage": 3.5,

    "ema_fast": 34,
    "ema_slow": 144,
    "ema_sep_trend": 0.0018,
    "ema_sep_chop": 0.0012,

    "donchian_period": 16,
    "breakout_confirm": 0.00055,
    "atr_period": 14,
    "adx_period": 14,
    "adx_trend_min": 16,
    "trail_atr_mult": 2.2,
    "trend_tp_atr_mult": 2.9,
    "max_hold_bars": 84,

    "bb_period": 20,
    "bb_std": 2.7,
    "rsi_period": 14,
    "rsi_long_max": 30,
    "rsi_short_min": 70,
    "mr_stop_atr_mult": 1.8,
    "mr_tp_mid": True,

    "min_atr_pct": 0.00075,
    "cooldown_bars_after_exit": 1,

    "dd_pause_pct": 0.14,
    "dd_pause_bars": 96,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 36,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
