#!/usr/bin/env python3
# Regime Strategy v1.1 (1H, BTC) - based on v1, tuned for more trades + better robustness

params = {
    "name": "run_btc_regime_1h_v1_1",
    "exchange": "bitget",
    "symbol": "BTC/USDT:USDT",
    "timeframe": "1h",
    "margin_mode": "isolated",

    # Exposure
    "position_size_percentage": 16,   # was conservative; slightly higher but still moderate
    "base_leverage": 2.0,
    "vol_lookback_hours": 168,             # keep leverage
    "use_longs": True,
    "use_shorts": True,

    # Regime detection (slightly more permissive -> more trades)
    "ema_fast": 34,
    "ema_slow": 144,
    "ema_sep_trend": 0.0018,          # lower threshold -> more bars counted as trend
    "ema_sep_chop": 0.0012,

    # Trend leg (tighter trailing, slightly earlier TP)
    "donchian_period": 20,
    "breakout_confirm": 0.0006,       # was likely higher; allow more breakouts
    "atr_period": 14,
    "adx_period": 14,
    "adx_trend_min": 18,
    "trail_atr_mult": 2.0,
    "trend_tp_atr_mult": 3.0,
    "max_hold_bars": 72,

    # Mean-reversion leg (keep but avoid micro-chop churn)
    "bb_period": 20,
    "bb_std": 2.5,
    "rsi_period": 14,
    "rsi_long_max": 32,
    "rsi_short_min": 68,
    "mr_stop_atr_mult": 1.6,
    "mr_tp_mid": True,                # take profit at mid-band

    # Filters to reduce fee-churn (important on 1H)
    "min_atr_pct": 0.00045,           # skip ultra-low vol hours
    "cooldown_bars_after_exit": 2,    # avoid immediate re-entry

    # Risk circuit breakers
    "dd_pause_pct": 0.12,
    "dd_pause_bars": 72,
    "loss_streak_pause_n": 3,
    "loss_streak_pause_bars": 24,

    # Fees (match your prior backtests)
    "maker_fee": 0.0002,
    "taker_fee": 0.0006,
}
