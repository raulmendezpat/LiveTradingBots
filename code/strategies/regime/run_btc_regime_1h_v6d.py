#!/usr/bin/env python3
# Regime Strategy v6c (1H, BTC, trend-only) - ensure trades + debug

params = {
    "name": "run_btc_regime_1h_v6d",
    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    "enable_trend": True,
    "enable_mr": False,

    "position_size_percentage": 14,
    "base_leverage": 2.0,
    "target_vol_annual": 1.05,
    "vol_lookback_hours": 168,
    "min_leverage": 1.0,
    "max_leverage": 4.0,

    # Relaxed filters
    "min_atr_pct": 0.00035,     # was 0.00045/0.0007
    "min_edge_pct": 0.0012,

    # Trend/chop proxy (more permissive)
    "ema_fast": 34,
    "ema_slow": 144,
    "ema_sep_trend": 0.0012,
    "ema_sep_chop": 0.0009,

    # HTF filter less restrictive
    "htf_ema_fast": 34,
    "htf_ema_slow": 144,
    "htf_neutral_band": 0.0060,

    # Breakout config
    "donchian_period": 14,
    "atr_period": 14,
    "trail_atr_mult": 2.2,
    "breakout_confirm": 0.0004,   # easier to trigger
    "trend_tp_atr_mult": 3.0,
    "max_hold_bars": 72,

    # MR unused
    "bb_period": 20,
    "bb_std": 2.6,
    "rsi_period": 14,
    "rsi_long_max": 30,
    "rsi_short_min": 70,
    "mr_stop_atr_mult": 1.6,

    "dd_pause_pct": 0.14,
    "dd_pause_bars": 72,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 24,
    "cooldown_bars_after_exit": 2,

    "use_longs": True,
    "use_shorts": True,

    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
