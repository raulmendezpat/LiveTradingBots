#!/usr/bin/env python3
# Regime Strategy v6 (1H, SOL) - balanced; includes diagnostics in backtest

params = {
    "name": "run_sol_regime_1h_v6",
    "exchange": "bitget",
    "symbol": "SOL/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    "enable_trend": True,
    "enable_mr": True,

    "position_size_percentage": 12,
    "base_leverage": 1.8,
    "target_vol_annual": 1.00,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 3.8,

    "min_atr_pct": 0.0010,
    "min_edge_pct": 0.0028,

    "ema_fast": 45,
    "ema_slow": 180,
    "ema_sep_trend": 0.0030,
    "ema_sep_chop": 0.0022,

    "htf_ema_fast": 50,
    "htf_ema_slow": 200,
    "htf_neutral_band": 0.0038,

    "donchian_period": 18,
    "atr_period": 14,
    "trail_atr_mult": 2.4,
    "breakout_confirm": 0.0012,
    "trend_tp_atr_mult": 3.2,
    "max_hold_bars": 84,

    "bb_period": 20,
    "bb_std": 2.7,
    "rsi_period": 14,
    "rsi_long_max": 30,
    "rsi_short_min": 70,
    "mr_stop_atr_mult": 1.7,

    "dd_pause_pct": 0.16,
    "dd_pause_bars": 96,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 36,
    "cooldown_bars_after_exit": 3,

    "use_longs": True,
    "use_shorts": True,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
